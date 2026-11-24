"""Test-time refinement (TTR) — the foundation model refined against the IMU targets.

Starts from the shared IMU init and targets (``vi3.grounding_methods.base``) and
fine-tunes the camera head, the depth head and the last ``unfreeze_n`` aggregator
blocks against the frame-0-relative IMU trajectory, with one learnable scalar α
absorbing the global metric scale (initialized from the IMU path length, trained
jointly). Loss: pose deltas + frame-0 anchor, plus a cross-view depth-consistency
term and a depth-to-init anchor that keeps the depth map's shape (and an optional
focal-length term). `TTRGrounder.DEFAULTS` is the paper configuration.

TTR adapts the model itself to the whole IMU trajectory, which also refines its
geometry; it is at its best with an accurate IMU trajectory.
"""

import torch

from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from vi3.data.preprocessing import prepare_batch
from vi3.grounding_methods.base import Grounder, _ScaleWrappedModel
from vi3.losses import (depth_consistency_loss, focal_length_loss,
                        origin_anchor_loss, pose_delta_loss)
from vi3.models.pi3_adapter import set_forward_intrinsics
from vi3.transforms import (derive_world_points_from_depth,
                            pose_enc_to_target_pose_torch,
                            target_pose_to_opencv_extrinsic)


class TTRGrounder(Grounder):
    """IMU-supervised test-time refinement with an explicit learnable scale α."""

    # The paper configuration; any key can be overridden through `config`.
    DEFAULTS = dict(
        steps=160,             # optimisation steps per window
        unfreeze_n=8,          # last N aggregator blocks trained (+ camera / depth heads)
        lr_agg=5e-6,           # aggregator blocks
        lr_cam=5e-6,           # camera head
        lr_depth=5e-6,         # depth head
        lr_alpha=1e-3,         # the scale α (one dof, so a larger rate)
        grad_clip=0.0,         # global grad-norm clipping (0 = off)
        # Loss weights (0 turns a term off).
        w_r=50.0,              # relative rotation
        w_t=10.0,              # relative translation
        w_anchor=1000.0,       # frame-0 pose at the origin
        w_depth_cons=10.0,     # cross-view depth consistency
        w_depth_anchor=1.0,    # depth stays close to its zero-shot shape
        w_fl=0.0,              # focal length (FoV channels of pose_enc) vs the dataset K
        # Depth-consistency inputs.
        warp_pose_source="imu",          # warp with "imu" target poses or the model's own ("feedforward")
        world_points_source="derived",   # unproject depth+pose ("derived") or the point-map head ("model")
    )

    def train(self, frame_indices, config):
        """Refine on `frame_indices`; return (grounded_model, history)."""
        return self._refine(frame_indices, {**self.DEFAULTS, **config})

    def _make_optimizer(self, model, config):
        """AdamW with one group each for the aggregator (`lr_agg`), the camera
        head (`lr_cam`), the depth head (`lr_depth`) and α (`lr_alpha`)."""
        depth_head = self._get_depth_head(model)
        depth_params = ([p for p in depth_head.parameters() if p.requires_grad]
                        if depth_head is not None else [])
        cam_params = list(model.camera_head.parameters())
        head_ids = {id(p) for p in depth_params + cam_params}
        agg_params = [p for p in model.parameters()
                      if p.requires_grad and id(p) not in head_ids]
        groups = [{"params": agg_params, "lr": config["lr_agg"]},
                  {"params": cam_params, "lr": config["lr_cam"]}]
        if depth_params:
            groups.append({"params": depth_params, "lr": config["lr_depth"]})
        groups.append({"params": [self.alpha], "lr": config["lr_alpha"]})
        return torch.optim.AdamW(groups)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _compute_loss(self, model_outputs, targets, config):
        """Pose deltas (relative translation L1 + relative rotation geodesic), the
        frame-0 origin anchor and the optional focal-length term."""
        pred_poses_raw = model_outputs["pose_enc"][0].float()  # bf16 -> fp32
        pred_target_pose = [pose_enc_to_target_pose_torch(p) for p in pred_poses_raw]

        t_loss, r_loss = pose_delta_loss(pred_target_pose, targets["p"], targets["q"])
        anchor = origin_anchor_loss(pred_target_pose[0])

        total = config["w_r"] * r_loss + config["w_t"] * t_loss + config["w_anchor"] * anchor
        components = {"t": t_loss.detach(), "r": r_loss.detach(),
                      "t_anchor": anchor.detach()}
        if config["w_fl"] > 0.0:
            H, W = model_outputs["depth"].shape[-3], model_outputs["depth"].shape[-2]
            fl = focal_length_loss(pred_poses_raw, targets["K"], (H, W))
            total = total + config["w_fl"] * fl
            components["fl"] = fl.detach()
        return total, components

    def _depth_losses(self, model_outputs, samples, targets, config):
        """Depth terms: the depth-to-init anchor, mean|depth − depth₀| with depth₀
        the model's depth at the first step, and the cross-view depth consistency:
        each frame's depth, warped into the other frames with the dataset
        intrinsics, must match their predicted depth."""
        device = self.device
        extras = {}
        extra_loss = torch.tensor(0.0, device=device)

        if config["w_depth_anchor"] > 0.0:
            d_now = model_outputs["depth"]
            if self._depth_anchor_ref is None:
                self._depth_anchor_ref = d_now.detach().clone()
            loss_a = (d_now - self._depth_anchor_ref).abs().mean()
            extras["depth_anchor"] = float(loss_a.detach())
            extra_loss = extra_loss + config["w_depth_anchor"] * loss_a
        if config["w_depth_cons"] <= 0.0:
            return extra_loss, extras

        images = torch.stack([s["image"] for s in samples], dim=0)  # [N, 3, H, W]
        H, W = images.shape[-2:]
        intr = torch.stack([s["K"].to(device) for s in samples])    # [N, 3, 3]
        if config["world_points_source"] == "model" and "world_points" in model_outputs:
            world_points = model_outputs["world_points"][0]         # [N, H, W, 3]
        else:
            world_points = derive_world_points_from_depth(
                model_outputs["depth"], model_outputs["pose_enc"], intr, (H, W),
            )[0]
        if config["warp_pose_source"] == "imu":
            extr = torch.stack([
                target_pose_to_opencv_extrinsic(targets["p"][i], targets["q"][i])
                for i in range(len(targets["p"]))
            ])                                                      # [N, 3, 4]
        elif config["warp_pose_source"] == "feedforward":
            extr = pose_encoding_to_extri_intri(
                model_outputs["pose_enc"], image_size_hw=(H, W))[0][0]
        else:
            raise ValueError(f"warp_pose_source must be 'imu' or 'feedforward', "
                             f"got {config['warp_pose_source']!r}")
        depths = model_outputs["depth"][0, ..., 0]                  # [N, H, W]
        loss_d = depth_consistency_loss(depths, world_points, extr, intr)
        extras["depth_c"] = float(loss_d.detach())
        extra_loss = extra_loss + config["w_depth_cons"] * loss_d
        return extra_loss, extras

    # ------------------------------------------------------------------
    # α initialization from the IMU trajectory (once, before step 0)
    # ------------------------------------------------------------------
    def _init_alpha_from_imu(self, model, samples, targets):
        """α₀ = Σ‖p_imu‖ / Σ‖p_pred‖, the path-length ratio of the IMU trajectory
        to the frozen model's (α = 1) camera centres. A scalar α only sets
        magnitude, never direction, so matching path length is enough."""
        with torch.no_grad():
            inner = model.model if isinstance(model, _ScaleWrappedModel) else model
            was_training = inner.training
            inner.eval()
            images = torch.stack([s["image"] for s in samples], dim=0).to(self.device)
            out = inner(images.unsqueeze(0))
            inner.train(was_training)

            pred = torch.stack([pose_enc_to_target_pose_torch(p)
                                for p in out["pose_enc"][0].float()])
            p_imu = torch.stack(targets["p"]).to(self.device)
            num = float(p_imu.norm(dim=-1).sum())
            den = float(pred[:, :3].norm(dim=-1).sum())
            s = num / den if den > 1e-12 else 1.0
            self.alpha.fill_(s)
        return s

    # ------------------------------------------------------------------
    # The refinement loop
    # ------------------------------------------------------------------
    def _refine(self, frame_indices, config):
        """Run TTR for `config['steps']` steps and return (grounded_model, history)."""
        # The window's IMU init and α — before the optimizer, which takes α.
        self._new_window(config)
        model = self.reset_model(unfreeze_n=config["unfreeze_n"], unfreeze_depth_head=True)
        optimizer = self._make_optimizer(model, config)

        prep = prepare_batch(self.dataset, frame_indices, self.device)
        images, samples = prep["images"], prep["samples"]
        self.cached_targets = self._build_targets(samples, config)
        self.cached_targets["K"] = torch.stack([s["K"].to(self.device) for s in samples])
        set_forward_intrinsics(model, self.cached_targets["K"],
                               (images.shape[-2], images.shape[-1]))

        s = self._init_alpha_from_imu(model, samples, self.cached_targets)
        print(f"             α initialized from the IMU trajectory: α₀ = {s:.4f}")

        history = {"total": [], "t": [], "r": [], "t_anchor": []}
        model.train()
        targets = self.cached_targets
        for step in range(config["steps"]):
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", dtype=self.dtype):
                out = model(images.unsqueeze(0))

            total_loss, components = self._compute_loss(out, targets, config)
            extra_loss, extras = self._depth_losses(out, samples, targets, config)
            total_loss = total_loss + extra_loss
            total_loss.backward()
            if config["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip"])
            optimizer.step()

            total_val = float(total_loss.detach())
            history["total"].append(total_val)
            for k, v in components.items():
                history.setdefault(k, []).append(float(v))
            for k, v in extras.items():
                history.setdefault(k, []).append(v)
            if step % 20 == 0:
                tail = "".join(f" | {k.upper()}: {float(v):.4f}" for k, v in components.items()
                               if k not in ("t", "r", "t_anchor"))
                tail += "".join(f" | {k.title()}: {v:.4f}" for k, v in extras.items())
                print(f"  [Step {step:03d}] Loss: {total_val:.4f} | "
                      f"R: {float(components['r']):.4f} | T: {float(components['t']):.4f} | "
                      f"Anchor: {float(components['t_anchor']):.4f}{tail}")
            self._post_step(step, components, history)

        return model, history
