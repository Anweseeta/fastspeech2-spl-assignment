#!/usr/bin/env python3
"""Plot training / validation loss curves from TensorBoard event files.

Reads:
    <Lang>/output/log/<dataset>/train/events.*   (tags Loss/*)
    <Lang>/output/log/<dataset>/val/events.*

Writes:
    Results/<Lang>/training_loss.png
    Results/<Lang>/validation_loss.png

Each figure shows total loss plus the mel / mel-postnet / pitch / energy /
duration components (the exact tags written by utils/tools.py::log).

Usage (run from repo root):
    python scripts/plot_losses.py --lang en
    python scripts/plot_losses.py --lang te
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LANGS = {
    "en": {"lang_dir": "English", "ds": "Svarah", "res": "English"},
    "te": {"lang_dir": "Telugu", "ds": "IndicTTS_Telugu", "res": "Telugu"},
}

TAGS = [
    ("Loss/total_loss", "total", "black", 2.0),
    ("Loss/mel_loss", "mel", "steelblue", 1.2),
    ("Loss/mel_postnet_loss", "mel postnet", "skyblue", 1.2),
    ("Loss/pitch_loss", "pitch", "seagreen", 1.2),
    ("Loss/energy_loss", "energy", "firebrick", 1.2),
    ("Loss/duration_loss", "duration", "darkorange", 1.2),
]


def read_series(log_dir):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    series = {}
    for tag, _, _, _ in TAGS:
        try:
            evts = ea.Scalars(tag)
            series[tag] = ([e.step for e in evts], [e.value for e in evts])
        except KeyError:
            print("  [warn] tag '{}' missing in {}".format(tag, log_dir))
    return series


def plot(series, title, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    # panel 1: total loss
    tag = "Loss/total_loss"
    if tag in series:
        steps, vals = series[tag]
        axes[0].plot(steps, vals, color="black", lw=1.5)
    axes[0].set_ylabel("total loss")
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)
    # panel 2: components
    for tag, label, color, lw in TAGS[1:]:
        if tag in series:
            steps, vals = series[tag]
            axes[1].plot(steps, vals, label=label, color=color, lw=lw)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("component losses")
    axes[1].legend(fontsize=8, ncol=3)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "te"], required=True)
    args = ap.parse_args()
    cfg = LANGS[args.lang]

    log_base = os.path.join(cfg["lang_dir"], "output", "log", cfg["ds"])
    out_base = os.path.join("Results", cfg["res"])
    os.makedirs(out_base, exist_ok=True)

    for split, fname in (("train", "training_loss.png"),
                         ("val", "validation_loss.png")):
        d = os.path.join(log_base, split)
        if not os.path.isdir(d):
            raise SystemExit("log dir missing: {} — train first".format(d))
        print("reading {}".format(d))
        series = read_series(d)
        if not series:
            raise SystemExit("no Loss/* scalars found in " + d)
        out = os.path.join(out_base, fname)
        plot(series, "{} loss ({}, {})".format(split, cfg["res"], cfg["ds"]), out)
        print("wrote " + out)


if __name__ == "__main__":
    main()
