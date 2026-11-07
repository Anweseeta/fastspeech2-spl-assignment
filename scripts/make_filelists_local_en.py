#!/usr/bin/env python3
"""Build English (Svarah) filelists from LOCAL parquet files (no streaming).

Svarah is gated and its only split is "test" (3 parquet files, ~1.1 GB).
The streaming path dies on flaky DNS, so this downloads the 3 files once
(curl, retry + resume, with your HF read token from the HF_TOKEN env var)
and then runs the exact same pass1/pass2 logic as make_filelists.py
from disk. No network is needed after the download.

Set your token in the terminal first (type it yourself, never paste it
in chat):
    export HF_TOKEN=hf_...

Usage (run from repo root):
    python scripts/make_filelists_local_en.py
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import make_filelists as mf

HF_ID = "ai4bharat/Svarah"
PARQUET_DIR = os.path.join("English", "parquet")
PARQUET_FILES = [
    "test-00000-of-00003.parquet",
    "test-00001-of-00003.parquet",
    "test-00002-of-00003.parquet",
]


def download_parquet(token):
    os.makedirs(PARQUET_DIR, exist_ok=True)
    for fname in PARQUET_FILES:
        url = ("https://huggingface.co/datasets/{}/resolve/main/data/{}"
               .format(HF_ID, fname))
        out = os.path.join(PARQUET_DIR, fname)
        if os.path.exists(out) and os.path.getsize(out) > 1000000:
            print("already have {}, skipping".format(fname))
            continue
        print("downloading {} ...".format(fname))
        r = subprocess.run([
            "curl", "-L", "--retry", "10", "--retry-all-errors",
            "--retry-delay", "5", "-C", "-",
            "-H", "Authorization: Bearer {}".format(token),
            "-o", out, url,
        ])
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("download failed for {}".format(fname))
        print("saved {} ({:.1f} MB)".format(fname, os.path.getsize(out) / 1e6))


def iter_local(hf_id, split, token):
    from datasets import load_dataset
    files = [os.path.join(PARQUET_DIR, f) for f in PARQUET_FILES]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    return load_dataset("parquet", data_files=files, split="train")


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN is not set. Run: export HF_TOKEN=hf_... "
            "(type your own token, never paste it in chat), then re-run.")

    cfg = dict(mf.LANGS["en"])
    # Svarah's only split is "test" (there is no train split) — record it
    # honestly in data_selection.md.
    cfg["split"] = "test"
    split = cfg["split"]
    out_root = os.path.join(cfg["lang_dir"], "raw_data", cfg["dir"])

    download_parquet(token)
    mf.iter_stream = iter_local

    print("pass 1: measuring durations from local parquet")
    probe = iter_local(cfg["hf_id"], split, None)
    cols = mf.detect_columns(probe)
    per_speaker, total_s = mf.pass1_collect(cfg["hf_id"], split, cols, None)

    selected, stats = mf.decide_split(per_speaker, cfg["train_h"], cfg["val_h"], total_s)

    print("pass 2: writing selected audio to {}".format(out_root))
    counts, hours = mf.pass2_write(cfg["hf_id"], split, cols, None,
                                   "en", selected, out_root)
    mf.write_data_selection_md(cfg, cols, total_s, stats, counts, hours)


if __name__ == "__main__":
    main()
