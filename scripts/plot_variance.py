#!/usr/bin/env python3
"""Plot variance-adaptor predictions for the assignment test sentences.

Reads the per-utterance dumps written by English/inference.py /
Telugu/inference.py:
    <Lang>/output/va/<dataset>/<ID>.npz        (duration_frames, duration_sec,
                                                pitch_hz, energy, ...)
    <Lang>/output/va/<dataset>/<ID>_phones.json (phone labels for the x-axis)

Writes, per test sentence:
    Results/<Lang>/variance_adaptor_plots/duration/<ID>.png
    Results/<Lang>/variance_adaptor_plots/pitch/<ID>.png
    Results/<Lang>/variance_adaptor_plots/energy/<ID>.png

Pitch/energy are plotted DENORMALIZED (Hz / raw energy) using the stats.json
values baked into the .npz by inference.py — i.e. what the variance adaptor
actually predicts, in interpretable units.

Usage (run from repo root):
    python scripts/plot_variance.py --lang en
    python scripts/plot_variance.py --lang te
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LANGS = {
    "en": {"lang_dir": "English", "ds": "Svarah", "res": "English"},
    "te": {"lang_dir": "Telugu", "ds": "IndicTTS_Telugu", "res": "Telugu"},
}


def _style(ax, title, xlabel):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=7, rotation=60)
    ax.grid(True, alpha=0.3)


def plot_duration(utt_id, phones, dur_frames, dur_sec, out_path):
    fig, ax = plt.subplots(figsize=(max(8, len(phones) * 0.35), 4))
    x = np.arange(len(phones))
    ax.bar(x, dur_frames, color="steelblue")
    ax.set_ylabel("frames (hop=256 @ 22050 Hz)")
    ax2 = ax.twinx()
    ax2.plot(x, dur_sec, "o-", color="darkorange", ms=3)
    ax2.set_ylabel("seconds", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(phones)
    _style(ax, "{}: predicted phone durations".format(utt_id), "phone")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_curve(utt_id, phones, values, ylabel, color, out_path, title):
    fig, ax = plt.subplots(figsize=(max(8, len(phones) * 0.35), 4))
    x = np.arange(len(phones))
    ax.plot(x, values, "o-", color=color, ms=4)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(phones)
    _style(ax, "{}: {}".format(utt_id, title), "phone")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "te"], required=True)
    args = ap.parse_args()
    cfg = LANGS[args.lang]

    va_dir = os.path.join(cfg["lang_dir"], "output", "va", cfg["ds"])
    out_base = os.path.join("Results", cfg["res"], "variance_adaptor_plots")
    for sub in ("duration", "pitch", "energy"):
        os.makedirs(os.path.join(out_base, sub), exist_ok=True)

    ids = sorted(f[:-4] for f in os.listdir(va_dir)
                 if f.endswith(".npz")) if os.path.isdir(va_dir) else []
    if not ids:
        raise SystemExit("no .npz dumps in {} — run inference.py first".format(va_dir))

    for utt_id in ids:
        d = np.load(os.path.join(va_dir, utt_id + ".npz"))
        with open(os.path.join(va_dir, utt_id + "_phones.json"),
                  encoding="utf-8") as f:
            phones = json.load(f)
        n = len(phones)
        dur_f = d["duration_frames"][:n]
        dur_s = d["duration_sec"][:n]
        pitch = d["pitch_hz"][:n]
        energy = d["energy"][:n]

        plot_duration(utt_id, phones, dur_f, dur_s,
                      os.path.join(out_base, "duration", utt_id + ".png"))
        plot_curve(utt_id, phones, pitch, "Hz", "seagreen",
                   os.path.join(out_base, "pitch", utt_id + ".png"),
                   "predicted pitch (F0)")
        plot_curve(utt_id, phones, energy, "energy", "firebrick",
                   os.path.join(out_base, "energy", utt_id + ".png"),
                   "predicted energy")
        print("plotted {}".format(utt_id))

    print("done: {} utterances -> {}".format(len(ids), out_base))


if __name__ == "__main__":
    main()
