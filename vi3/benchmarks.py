"""The evaluated sequences of the three datasets.

A `BenchCondition` describes one sequence: its dataset, the kwargs that build
the Dataset object, a label (`House/easy/P000`, `MH_01_easy`,
`indoor_forward_3`), a group and optional extras (difficulty, environment).
`SUITES` names the full datasets (`tartanair_full`, `euroc_full`,
`uzhfpv_full`, `all`) for `scripts/run.py --suite`.
"""

from dataclasses import dataclass
from itertools import product

from vi3.data import (EuRoCRefinementDataset, TartanAirRefinementDataset,
                       UZHFPVRefinementDataset)
from vi3.data.euroc import EUROC_SEQUENCES
from vi3.data.uzhfpv import UZHFPV_SEQUENCES
from vi3.paths import EUROC_ROOT, TARTANAIR_ROOT, UZHFPV_ROOT


# ---------------------------------------------------------------------------
# Condition descriptor
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchCondition:
    """One row of the benchmark.

    dataset:        "tartanair" | "euroc" | "uzhfpv"
    label:          short human-readable id (e.g. "House/easy/P000", "MH_01_easy")
    fields:         tuple of (name, value) used as kwargs to the Dataset ctor.
                    tuple-of-tuples instead of dict so the dataclass stays
                    hashable / frozen.
    group:          summary-grouping tag — TartanAir env name; EuRoC MH/V1/V2.
    extra:          tuple of (name, value) tags for per-subset breakdowns
                    (difficulty, env, ...). Same hashability trick.
    """
    dataset:  str
    label:    str
    fields:   tuple
    group:    str = ""
    extra:    tuple = ()

    def kwargs(self):
        return dict(self.fields)

    def tags(self):
        return dict(self.extra)

    def build_dataset(self, model_spec=None, frame_stride=1):
        """Construct the appropriate Dataset instance for this condition.

        `model_spec` (a `vi3.models.ModelSpec`) routes the dataset's
        preprocessing to match the active model backend. Defaults to
        VGGT inside the dataset class if omitted.

        All datasets are evaluated over their full GT-covered window —
        no stationary-prefix skip.
        """
        if self.dataset == "tartanair":
            return TartanAirRefinementDataset(
                TARTANAIR_ROOT, mode="pad",
                model_spec=model_spec, frame_stride=frame_stride, **self.kwargs(),
            )
        if self.dataset == "euroc":
            return EuRoCRefinementDataset(
                EUROC_ROOT, mode="pad",
                model_spec=model_spec, frame_stride=frame_stride, **self.kwargs(),
            )
        if self.dataset == "uzhfpv":
            return UZHFPVRefinementDataset(
                UZHFPV_ROOT, mode="pad",
                model_spec=model_spec, frame_stride=frame_stride, **self.kwargs(),
            )
        raise ValueError(f"Unknown dataset {self.dataset!r}")


# ---------------------------------------------------------------------------
# TartanAir registry
# ---------------------------------------------------------------------------

_TARTANAIR_ENVS         = ["House", "Hospital", "ModernCityDowntown", "ForestEnv"]
_TARTANAIR_DIFFICULTIES = ["easy", "hard"]
# Two trajectories per (env, difficulty): the first two IDs present in EVERY
# env/difficulty (ForestEnv has no P001, so the first common pair is P000/P002).
_TARTANAIR_TRAJS        = ["P000", "P002"]


def _make_tartanair_conditions():
    out = []
    for env, diff, traj in product(_TARTANAIR_ENVS, _TARTANAIR_DIFFICULTIES,
                                    _TARTANAIR_TRAJS):
        out.append(BenchCondition(
            dataset="tartanair",
            label=f"{env}/{diff}/{traj}",
            fields=(("env", env), ("difficulty", diff), ("traj", traj)),
            group=env,
            extra=(("difficulty", diff), ("env", env)),
        ))
    return out


TARTANAIR_ALL = _make_tartanair_conditions()


# ---------------------------------------------------------------------------
# EuRoC registry
# ---------------------------------------------------------------------------

def _euroc_difficulty(sequence_label):
    """Extract the difficulty tag from the sequence label."""
    for tag in ("easy", "medium", "difficult"):
        if sequence_label.endswith(tag):
            return tag
    return "unknown"


def _make_euroc_conditions():
    out = []
    for seq, (_category, group) in EUROC_SEQUENCES.items():
        out.append(BenchCondition(
            dataset="euroc",
            label=seq,
            fields=(("sequence", seq),),
            group=group,
            extra=(("difficulty", _euroc_difficulty(seq)),),
        ))
    return out


EUROC_ALL = _make_euroc_conditions()


# ---------------------------------------------------------------------------
# UZH-FPV registry
# ---------------------------------------------------------------------------

def _uzhfpv_difficulty(sequence_label):
    """Coarse difficulty tag for a UZH-FPV sequence.

    UZH-FPV has no published difficulty tier per sequence — we use the
    family + outdoor/indoor as a proxy: indoor_forward < indoor_45 <
    outdoor_*. Useful only for breakdown plots; not part of the dataset's
    official categorization.
    """
    if sequence_label.startswith("outdoor"):
        return "outdoor"
    if sequence_label.startswith("indoor_45"):
        return "indoor_45"
    if sequence_label.startswith("indoor_forward"):
        return "indoor_forward"
    return "unknown"


def _make_uzhfpv_conditions():
    out = []
    for seq, (_family, group) in UZHFPV_SEQUENCES.items():
        out.append(BenchCondition(
            dataset="uzhfpv",
            label=seq,
            fields=(("sequence", seq),),
            group=group,
            extra=(("difficulty", _uzhfpv_difficulty(seq)),),
        ))
    return out


UZHFPV_ALL = _make_uzhfpv_conditions()


# The demo cell: what a bare `scripts/run.py` runs and the README shows.
# Its window 1 (of the 3 evaluated) is a clean example of metric grounding.
DEMO_CONDITION = "House/easy/P000"
DEMO_WINDOW = 1


# ---------------------------------------------------------------------------
# Named suites
# ---------------------------------------------------------------------------

SUITES = {
    "tartanair_full": TARTANAIR_ALL,
    "euroc_full":     EUROC_ALL,
    "uzhfpv_full":    UZHFPV_ALL,
    "all":            TARTANAIR_ALL + EUROC_ALL + UZHFPV_ALL,
}


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------

def lookup_condition(label):
    """Find one condition by its label across all datasets.

    Raises KeyError if not found.
    """
    for c in TARTANAIR_ALL + EUROC_ALL + UZHFPV_ALL:
        if c.label == label:
            return c
    raise KeyError(
        f"No condition with label {label!r}. "
        f"Known TartanAir labels: {[c.label for c in TARTANAIR_ALL][:3]}... "
        f"Known EuRoC labels: {[c.label for c in EUROC_ALL][:3]}... "
        f"Known UZH-FPV labels: {[c.label for c in UZHFPV_ALL][:3]}..."
    )


def resolve_conditions(suite=None, conditions=None):
    """Pick a list of `BenchCondition` from CLI-style args.

    Exactly one of `suite` or `conditions` must be provided.
    `suite`: a key from SUITES.
    `conditions`: comma-separated list of labels, e.g. "MH_01_easy,V1_01_easy".
    """
    if (suite is None) == (conditions is None):
        raise ValueError("Pass exactly one of `suite` or `conditions`.")
    if suite is not None:
        if suite not in SUITES:
            raise KeyError(f"Unknown suite {suite!r}. Known: {sorted(SUITES.keys())}")
        return SUITES[suite]
    return [lookup_condition(c.strip()) for c in conditions.split(",") if c.strip()]
