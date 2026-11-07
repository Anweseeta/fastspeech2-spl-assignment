#!/usr/bin/env python3
"""Build Telugu filelists from LOCAL parquet files (no streaming).

The streaming path keeps dying on flaky DNS, so this downloads the 4
IndicTTS_Telugu parquet files once (curl, retry + resume) and then runs
the exact same pass1/pass2 logic as make_filelists.py from disk.
No network is needed after the download.

Usage (run from repo root):
    python scripts/make_filelists_local.py --lang te --only_speaker 0
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import make_filelists as mf

HF_ID = "SPRINGLab/IndicTTS_Telugu"
COMMIT = "7ab3c18491bef3f5d4e820c2701b5f78f54031d6"
PARQUET_DIR = os.path.join("Telugu", "parquet")


def download_parquet():
    os.makedirs(PARQUET_DIR, exist_ok=True)
    for i in range(4):
        fname = "train-0000{}-of-00004.parquet".format(i)
        url = ("https://huggingface.co/datasets/{}/resolve/{}/data/{}"
               .format(HF_ID, COMMIT, fname))
        out = os.path.join(PARQUET_DIR, fname)
        if os.path.exists(out) and os.path.getsize(out) > 1000000:
            print("already have {}, skipping".format(fname))
            continue
        print("downloading {} ...".format(fname))
        r = subprocess.run([
            "curl", "-L", "--retry", "10", "--retry-all-errors",
            "--retry-delay", "5", "-C", "-", "-o", out, url,
        ])
        if r.returncode != 0 or not os.path.exists(out):
            raise RuntimeError("download failed for {}".format(fname))
        print("saved {} ({:.1f} MB)".format(fname, os.path.getsize(out) / 1e6))


def iter_local(hf_id, split, token):
    from datasets import load_dataset
    files = [os.path.join(PARQUET_DIR, "train-0000{}-of-00004.parquet".format(i))
             for i in range(4)]
    for f in files:
        if not os.path.exists(f):
            raise FileNotFoundError(f)
    return load_dataset("parquet", data_files=files, split="train")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "te"], default="te")
    ap.add_argument("--only_speaker", default=None)
    args = ap.parse_args()

    cfg = mf.LANGS[args.lang]
    split = cfg["split"]
    out_root = os.path.join(cfg["lang_dir"], "raw_data", cfg["dir"])

    download_parquet()
    mf.iter_stream = iter_local

    print("pass 1: measuring durations from local parquet")
    probe = iter_local(cfg["hf_id"], split, None)
    cols = mf.detect_columns(probe)
    per_speaker, total_s = mf.pass1_collect(cfg["hf_id"], split, cols, None)

    if args.only_speaker is not None:
        keep = args.only_speaker
        per_speaker = {k: v for k, v in per_speaker.items() if k == keep}
        if not per_speaker:
            raise ValueError("speaker {!r} not found".format(keep))
        total_s = sum(d for utts in per_speaker.values() for _, d, _ in utts)
        print("restricted to speaker {}: {:.2f}h pool".format(keep, total_s / 3600))

    selected, stats = mf.decide_split(per_speaker, cfg["train_h"], cfg["val_h"], total_s)

    print("pass 2: writing selected audio to {}".format(out_root))
    counts, hours = mf.pass2_write(cfg["hf_id"], split, cols, None,
                                   args.lang, selected, out_root)
    mf.write_data_selection_md(cfg, cols, total_s, stats, counts, hours)


if __name__ == "__main__":
    main()
