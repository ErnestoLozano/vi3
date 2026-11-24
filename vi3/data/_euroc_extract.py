"""Idempotent on-demand unzipper for EuRoC sequence archives.

ASL ships each sequence as `<category>/<sequence>/<sequence>.zip` (a few GB
each). We extract into `<category>/<sequence>/mav0/...` lazily — the dataset
class calls `ensure_extracted(...)` on first access, and we only unzip if
the expected tree is missing.
"""

import os
import zipfile


def ensure_extracted(sequence_dir):
    """Extract `<sequence_dir>/<seq>.zip` if `<sequence_dir>/mav0` is missing.

    sequence_dir: path like '<EUROC_ROOT>/machine_hall/MH_01_easy'.

    Returns the path to the extracted `mav0/` directory. Raises
    FileNotFoundError if neither the zip nor the mav0 tree exist.
    """
    mav0 = os.path.join(sequence_dir, "mav0")
    if os.path.isdir(mav0):
        return mav0

    # Find <seq>.zip in the directory. EuRoC also ships a .bag we don't use.
    if not os.path.isdir(sequence_dir):
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    zips = [f for f in os.listdir(sequence_dir) if f.endswith(".zip")]
    if not zips:
        raise FileNotFoundError(
            f"Neither {mav0!r} nor a .zip archive in {sequence_dir!r}"
        )
    if len(zips) > 1:
        raise RuntimeError(f"Multiple .zip files in {sequence_dir!r}: {zips}")

    zip_path = os.path.join(sequence_dir, zips[0])
    print(f"[euroc] extracting {zip_path} -> {sequence_dir}/mav0/ ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(sequence_dir)
    if not os.path.isdir(mav0):
        raise RuntimeError(
            f"Extraction completed but {mav0!r} still missing — archive layout unexpected"
        )
    return mav0
