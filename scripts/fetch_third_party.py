#!/usr/bin/env python3
"""Fetch the third-party code VI3 does not redistribute, at a fixed commit, into third_party/.

    python scripts/fetch_third_party.py

  * VGGT-Ω (https://github.com/facebookresearch/vggt-omega), FAIR Noncommercial Research License: the
    repository at the commit below (without its example images), into third_party/vggt-omega; then
    `pip install -e third_party/vggt-omega` (see the README). Its checkpoint is gated on the Hugging Face hub and
    downloads separately.

Standard library only (it runs before VI3 is installed). Re-running skips what is already there.
"""
import os
import shutil
import tarfile
import tempfile
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# name -> (archive URL at the pinned commit, destination under third_party/, top-level entries left out)
FETCH = {
    "VGGT-Ω": ("https://github.com/facebookresearch/vggt-omega/archive/"
               "282ec70363edeff59424bf43731658092fba3d37.tar.gz", "vggt-omega", ("examples", ".gitignore")),
}


def fetch(url, dst, skip=()):
    """Extract the archive at `url` (one top-level folder, as GitHub's) into `dst`, without the entries `skip`."""
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "src.tar.gz")
        urllib.request.urlretrieve(url, arc)
        with tarfile.open(arc) as tf:
            top = {m.name.split("/", 1)[0] for m in tf.getmembers()}
            assert len(top) == 1, f"unexpected archive layout: {sorted(top)[:5]}"
            tf.extractall(tmp, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        src = os.path.join(tmp, top.pop())
        for s in skip:
            p = os.path.join(src, s)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
        shutil.move(src, dst)


def main():
    for name, (url, sub, skip) in FETCH.items():
        dst = os.path.join(REPO_ROOT, "third_party", sub)
        if os.path.exists(dst):
            print(f"{name}: {dst} exists, skipped")
            continue
        print(f"{name} -> {dst}", flush=True)
        fetch(url, dst, skip)
    print("done")


if __name__ == "__main__":
    main()
