"""Download UCI SECOM, verify/write checksum manifest, load in time order."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.json"
FILES = ["secom.data", "secom_labels.data", "secom.names"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    z = RAW / "secom.zip"
    if force or not z.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, z)
    with zipfile.ZipFile(z) as zf:
        zf.extractall(RAW)
    hashes = {f: sha256(RAW / f) for f in ["secom.zip"] + FILES}
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text())
        for f, h in hashes.items():
            if old["sha256"].get(f) != h:
                raise SystemExit(f"CHECKSUM MISMATCH for {f}: manifest {old['sha256'].get(f)} vs local {h}")
        print("checksums match data/MANIFEST.json")
        return
    X, y, t = _read_raw()
    man = {
        "source": "UCI Machine Learning Repository, SECOM (id 179)",
        "landing_page": "https://archive.ics.uci.edu/dataset/179/secom",
        "url": URL,
        "downloaded_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashes,
        "rows": int(X.shape[0]),
        "sensor_cols": int(X.shape[1]),
        "label_counts": {"pass(-1)": int((y == -1).sum()), "fail(1)": int((y == 1).sum())},
        "timestamp_min": str(t.min()),
        "timestamp_max": str(t.max()),
        "license": "CC BY 4.0 (per UCI listing)",
    }
    MANIFEST.write_text(json.dumps(man, indent=2) + "\n")
    print(f"wrote {MANIFEST}")


def _read_raw():
    X = pd.read_csv(RAW / "secom.data", sep=r"\s+", header=None, na_values=["NaN"])
    X.columns = [f"sensor_{i:03d}" for i in range(X.shape[1])]
    lab = pd.read_csv(RAW / "secom_labels.data", sep=" ", header=None, names=["label", "ts"], quotechar='"')
    t = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    return X, lab["label"].to_numpy(), t


def load():
    """Return (X DataFrame, y in {0 pass,1 fail}, timestamps), sorted by time with a STABLE sort (ties keep file order)."""
    X, y, t = _read_raw()
    order = np.argsort(t.to_numpy(), kind="mergesort")
    X = X.iloc[order].reset_index(drop=True)
    y = (y[order] == 1).astype(int)
    t = t.iloc[order].reset_index(drop=True)
    return X, y, t, order


if __name__ == "__main__":
    download()
    X, y, t, order = load()
    print(X.shape, "fails", y.sum(), "already time-ordered:", bool((order == np.arange(len(order))).all()),
          "tied timestamps:", int(t.duplicated().sum()))
