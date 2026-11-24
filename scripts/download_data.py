#!/usr/bin/env python3
"""Download the evaluation datasets from their official sources into the layout
the VI3 loaders read (see vi3/data/README.md).

    python scripts/download_data.py euroc                    # all 11 sequences
    python scripts/download_data.py uzhfpv --sequences indoor_forward_5,outdoor_45_1
    python scripts/download_data.py tartanair --sequences House/easy   # `pip install tartanair -c requirements.txt`
    python scripts/download_data.py all

Each dataset goes to `data/<name>` in the repo, or to its root (`EUROC_ROOT`,
`UZHFPV_ROOT`, `TARTANAIR_ROOT`, or `--root`). Re-running skips what is already there.

  * EuRoC MAV — ETH Research Collection (doi:10.3929/ethz-b-000690084). The data
    ships as one archive per room; only the per-sequence ASL zips are fetched from
    it (HTTP range requests, skipping the ROS bags), to
    `<root>/<room>/<seq>/<seq>.zip`. The loader unpacks them on first use.
  * UZH-FPV — fpv.ifi.uzh.ch (v3 release): the Snapdragon zips with ground truth
    and the per-family calibration, extracted to `<root>/<seq>/snapdragon/` and
    `<root>/calib/<family>/`.
  * TartanAir V2 — the official `tartanair` package: left front camera images,
    depth and IMU of the evaluated environments, all their trajectories. Name an
    environment (`House`) or one of its difficulties (`House/easy`).

The datasets keep their own licenses and terms of use; see their websites.
"""
import argparse
import os
import shutil
import struct
import urllib.request
import zipfile
import zlib

from vi3.data.euroc import EUROC_SEQUENCES
from vi3.data.uzhfpv import UZHFPV_SEQUENCES
from vi3.paths import EUROC_ROOT, TARTANAIR_ROOT, UZHFPV_ROOT

_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
_CHUNK = 1 << 22

# EuRoC: Research Collection bitstreams, one archive per room.
_EUROC_ARCHIVES = {
    "machine_hall": "7b2419c1-62b5-4714-b7f8-485e5fe3e5fe",
    "vicon_room1":  "02ecda9a-298f-498b-970c-b7c44334d880",
    "vicon_room2":  "ea12bc01-3677-4b4c-853d-87c7870b8c44",
}
_EUROC_URL = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{}/content"
_UZHFPV_SEQ_URL = ("https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv-newer-versions/"
                   "v3/{}_snapdragon_with_gt.zip")
_UZHFPV_CALIB_URL = ("https://download.ifi.uzh.ch/rpg/web/datasets/uzh-fpv/calib/"
                     "{}_calib_snapdragon.zip")
# TartanAir: the environments, difficulties and trajectories VI3 evaluates.
_TARTANAIR_ENVS = ["House", "Hospital", "ModernCityDowntown", "ForestEnv"]
_TARTANAIR_DIFFICULTIES = ["easy", "hard"]


def _request(url, start=None, end=None):
    headers = {"User-Agent": _UA}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def _copy(src, dst_path, total, label, decompress=None):
    """Stream `src` to `dst_path` (via a .part file), optionally inflating it."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    part, done = dst_path + ".part", 0
    with open(part, "wb") as f:
        while True:
            chunk = src.read(_CHUNK)
            if not chunk:
                break
            done += len(chunk)
            f.write(decompress.decompress(chunk) if decompress else chunk)
            print(f"\r  {label}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", flush=True)
        if decompress:
            f.write(decompress.flush())
    print()
    os.replace(part, dst_path)


def download(url, dst_path):
    """Download `url` to `dst_path` unless it is already there."""
    if os.path.exists(dst_path):
        return
    with _request(url) as r:
        _copy(r, dst_path, int(r.headers.get("Content-Length", 0)), os.path.basename(dst_path))


class RemoteZip:
    """Reads members of a zip over HTTP without downloading the whole archive."""

    def __init__(self, url):
        self.url = url
        with _request(url, 0, 0) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        tail = self._read(max(0, self.size - (1 << 18)), self.size - 1)
        eocd = tail.rfind(b"PK\x05\x06")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
        if cd_off == 0xFFFFFFFF:                                     # zip64
            eocd64 = tail.rfind(b"PK\x06\x06")
            cd_size, cd_off = struct.unpack("<QQ", tail[eocd64 + 40:eocd64 + 56])
        self.members = self._parse_central_directory(self._read(cd_off, cd_off + cd_size - 1))

    def _read(self, start, end):
        with _request(self.url, start, end) as r:
            return r.read()

    @staticmethod
    def _parse_central_directory(cd):
        members, p = {}, 0
        while cd[p:p + 4] == b"PK\x01\x02":
            method, = struct.unpack("<H", cd[p + 10:p + 12])
            csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
            n, e, c = struct.unpack("<HHH", cd[p + 28:p + 34])
            offset, = struct.unpack("<I", cd[p + 42:p + 46])
            name = cd[p + 46:p + 46 + n].decode("utf-8")
            extra, q = cd[p + 46 + n:p + 46 + n + e], 0
            while q + 4 <= len(extra):                                 # zip64 extra field
                tag, ln = struct.unpack("<HH", extra[q:q + 4])
                if tag == 0x0001:
                    vals, k = list(struct.unpack(f"<{ln // 8}Q", extra[q + 4:q + 4 + ln // 8 * 8])), 0
                    if usize == 0xFFFFFFFF:
                        usize, k = vals[k], k + 1
                    if csize == 0xFFFFFFFF:
                        csize, k = vals[k], k + 1
                    if offset == 0xFFFFFFFF:
                        offset = vals[k]
                q += 4 + ln
            members[name] = (offset, csize, usize, method)
            p += 46 + n + e + c
        return members

    def extract(self, name, dst_path):
        """Stream member `name` to `dst_path` unless it is already there."""
        offset, csize, usize, method = self.members[name]
        if os.path.exists(dst_path) and os.path.getsize(dst_path) == usize:
            return
        header = self._read(offset, offset + 29)
        n, e = struct.unpack("<HH", header[26:30])
        start = offset + 30 + n + e
        with _request(self.url, start, start + csize - 1) as r:
            _copy(r, dst_path, csize, os.path.basename(dst_path),
                  zlib.decompressobj(-15) if method == 8 else None)


def euroc(root, sequences):
    by_room = {}
    for seq in sequences:
        by_room.setdefault(EUROC_SEQUENCES[seq][0], []).append(seq)
    for room, seqs in by_room.items():
        print(f"[euroc] {room}: {', '.join(seqs)}")
        archive = RemoteZip(_EUROC_URL.format(_EUROC_ARCHIVES[room]))
        for seq in seqs:
            archive.extract(f"{room}/{seq}/{seq}.zip", os.path.join(root, room, seq, f"{seq}.zip"))


def _unzip(zip_path, dst_dir, strip_top=False):
    """Extract `zip_path` into `dst_dir`, optionally dropping its top-level folder."""
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            name = m.filename.split("/", 1)[-1] if strip_top else m.filename
            if not name or m.is_dir():
                continue
            out = os.path.join(dst_dir, name)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with zf.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)


def uzhfpv(root, sequences, keep_zips):
    zips = os.path.join(root, "_zips")
    for family in sorted({UZHFPV_SEQUENCES[s][0] for s in sequences}):
        dst = os.path.join(root, "calib", family)
        if os.path.isdir(dst):
            continue
        print(f"[uzhfpv] calibration: {family}")
        zip_path = os.path.join(zips, f"{family}_calib_snapdragon.zip")
        download(_UZHFPV_CALIB_URL.format(family), zip_path)
        _unzip(zip_path, dst, strip_top=True)
    for seq in sequences:
        dst = os.path.join(root, seq, "snapdragon")
        if os.path.isfile(os.path.join(dst, "groundtruth.txt")):
            continue
        print(f"[uzhfpv] {seq}")
        zip_path = os.path.join(zips, f"{seq}_snapdragon_with_gt.zip")
        download(_UZHFPV_SEQ_URL.format(seq), zip_path)
        _unzip(zip_path, dst)
        if not keep_zips:
            os.remove(zip_path)
    if not keep_zips:
        shutil.rmtree(zips, ignore_errors=True)


def tartanair(root, envs):
    try:
        import tartanair as ta
    except ImportError:
        raise SystemExit("TartanAir is downloaded with its official package: "
                         "`pip install tartanair -c requirements.txt`.")
    ta.init(root)
    for name in envs:
        env, _, diff = name.partition("/")
        ta.download(env=[env], difficulty=[diff] if diff else _TARTANAIR_DIFFICULTIES,
                    modality=["image", "depth", "imu"], camera_name=["lcam_front"],
                    unzip=True, delete_zip=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["euroc", "uzhfpv", "tartanair", "all"])
    ap.add_argument("--root", default=None,
                    help="destination (default: the dataset's env var / vi3.paths root)")
    ap.add_argument("--sequences", default=None,
                    help="comma-separated EuRoC / UZH-FPV sequences or TartanAir "
                         "environments[/difficulty] (default: all that VI3 evaluates)")
    ap.add_argument("--keep_zips", action="store_true",
                    help="UZH-FPV: keep the downloaded zips after extraction")
    a = ap.parse_args()
    if a.dataset == "all" and (a.root or a.sequences):
        ap.error("--root / --sequences apply to a single dataset")

    def pick(known):
        names = [s.strip() for s in a.sequences.split(",")] if a.sequences else list(known)
        unknown = [s for s in names if s not in known]
        if unknown:
            ap.error(f"unknown {a.dataset} sequences {unknown}; known: {sorted(known)}")
        return names

    if a.dataset in ("euroc", "all"):
        euroc(a.root or EUROC_ROOT, pick(EUROC_SEQUENCES))
    if a.dataset in ("uzhfpv", "all"):
        uzhfpv(a.root or UZHFPV_ROOT, pick(UZHFPV_SEQUENCES), a.keep_zips)
    if a.dataset in ("tartanair", "all"):
        tartanair(a.root or TARTANAIR_ROOT, pick(
            _TARTANAIR_ENVS + [f"{e}/{d}" for e in _TARTANAIR_ENVS
                               for d in _TARTANAIR_DIFFICULTIES]))


if __name__ == "__main__":
    main()
