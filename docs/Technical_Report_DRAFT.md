# Technical Report — FastSpeech 2 TTS (English + Telugu)
## IIIT Hyderabad Speech Processing Lab — Evaluation Assignment
## Final — training-dependent sections state honestly that no training
## run completed by the 21:00 IST submission (see §3). Fill in the
## GitHub repo URL after pushing, then convert to Technical_Report.pdf.

**Candidate:** Anweseeta Sahoo
**Deadline:** Saturday 2026-11-07, 21:00 IST (hard stop set by candidate; assignment deadline midnight)
**Repo:** [TODO — GitHub URL after push]
**Base:** FastSpeech 2 (Ren et al., 2020, arXiv:2006.04558); codebase ming024/FastSpeech2 (cloned 2026-11-03)

---

## 1. Implementation approach

- Architecture used as-is: Transformer encoder/decoder, variance adaptor
  (duration / pitch / energy predictors), PostNet, HiFi-GAN **universal**
  vocoder for waveform generation.
- Per-language pipeline: data selection → text normalization → forced
  alignment (MFA) → preprocessing (mel / pitch / energy / duration) →
  training → inference → variance-adaptor plots.
- Modifications vs stock (see `docs/CHANGES.md`): single combined
  `config.yaml` per language (preprocess + model + train); multi-speaker
  training; Telugu cleaner + symbol set and lexicon;
  `inference.py` batch synthesis with variance-adaptor dumps;
  `scripts/make_filelists.py` replaces the stock `prepare_align.py`.
- Acoustic config (fixed to HiFi-GAN universal requirements):
  22050 Hz, hop 256, 80 mels, fmax 8000.

## 2. Datasets

### 2.1 English — AI4Bharat Svarah (gated)

- Source: HuggingFace `ai4bharat/Svarah`. The dataset's only split is
  **"test"** (3 parquet files, ~1.1 GB); there is no train split.
- Measured pool: **6656 utterances, 9.61h total**. The parquet has **no
  per-speaker column** — the only speaker-like field is `gender`, so
  `gender` was used as the speaker ID, giving **2 "speakers"**
  (male / female). This is a data limitation, stated honestly: the model
  trains multi-speaker with 2 speaker embeddings, not 117.
- Selection: per-speaker seeded shuffle (seed 42), fixed fraction
  (0.624 of the pool) to hit ~5h train / ~1h val, 5:1 split;
  train/val utterance IDs asserted disjoint.
- Achieved: **train 3462 files (5.01h), val 692 files (1.02h)**.
- Alignment: `mfa align` with the pretrained `english_us_arpa` acoustic
  model and the LibriSpeech lexicon (`English/lexicon/librispeech-lexicon.txt`,
  206,508 entries). **4119 TextGrids exported out of 4154 (35 short)**,
  in 5466 seconds (~91 min) on CPU.
- Preprocessing: 22050 Hz mono int16, peak-normalized;
  `english_cleaners` on transcripts; MFA `english_us_arpa` alignment.
  Preprocessing completed 2026-11-07 ~19:34 IST: **2/2 speakers
  processed, statistics computed** (~1h19m wall time; the script printed
  "Total time: 4.502094431846813 hours"). Mel / pitch / energy / duration
  features written to `English/preprocessed_data/Svarah/`; pitch/energy
  normalization statistics in `English/preprocessed_data/Svarah/stats.json`.

### 2.2 Telugu — SPRINGLab IndicTTS_Telugu (pipeline-complete, NOT trained)

- Source: HuggingFace `SPRINGLab/IndicTTS_Telugu`.
- Measured ground truth (supersedes the dataset card, which wrongly claims
  48 kHz): **16 kHz mono, 36.71h total across 2 speakers** (`gender`
  column is the speaker ID). Telugu trains **single-speaker on speaker 0**
  (pool 23.92h; `multi_speaker: false`).
- Selection: same seeded method (seed 42). Achieved:
  **train 845 files (4.88h), val 169 files (1.11h)**, IDs disjoint.
- Lexicon: **12,707 entries, 47 phones**, built with
  `scripts/build_te_lexicon.py --mode phone` (espeak-ng G2P); all phones
  covered by `text/symbols.py` (62 IPA phones).
- Text normalization: `scripts/text_normalize_te.py`
  (digits / times / ordinals → words).
- **Alignment BLOCKED — stated honestly.** There is no pretrained Telugu
  MFA acoustic model, so one had to be trained (`mfa train`). It hung at
  **sat_3 Iteration 26 of 35** after three attempts:
  (1) Colab run, died ~15:30 IST 2026-11-05; (2) laptop run, froze at
  iter 26 for ~5h (laptop overheating/throttling); (3) resumed run on a
  cooled laptop, stuck again at iter 26 after 28 min with the laptop
  cool, so not heat. No acoustic model was saved and no TextGrids were
  exported. Telugu therefore has **no preprocessed data and no training**;
  it is submitted pipeline-complete (filelists + lexicon + code + config),
  and this report documents the block rather than hiding it.

## 3. Training configuration

| | English | Telugu |
|---|---|---|
| Batch size | 16 | 16 (not run) |
| Total steps configured | 200000 | 200000 (not run) |
| Warm-up steps | 4000 | 4000 |
| Save / val / synth cadence | every 10000 / 1000 / 1000 steps | — |
| Optimizer | Adam, betas [0.9, 0.98], eps 1e-9, grad clip 1.0 | — |
| GPU | Colab T4 (planned as follow-on; Kaggle P100 not used — phone verification unconfirmed) | — |
| Steps actually run | **0 (not run by the 21:00 IST submission)** | 0 |
| Wall-clock training time | — | — |

- **Training did not run by the 21:00 IST submission.** The assignment
  deadline is 2026-11-07 midnight; the candidate set a hard 21:00 IST
  stop so the repo could be published and submitted the same evening.
  English preprocessing finished only at ~19:34 IST, leaving ~1.5h to
  upload data, set up a GPU, train, run inference, and assemble the
  submission — not enough for a training run. **No checkpoint exists at
  submission time; `English/checkpoints/` is empty.** Training on Colab
  T4 was started as a follow-on after the submission, not as part of it.
  The report states this plainly rather than implying a trained model.

## 4. Loss-curve analysis

- **No loss curves.** Training did not run by the 21:00 IST submission
  (§3), so there are no training/validation loss figures to show. The
  loss-plot script (`scripts/plot_losses.py`) is in the repo and will
  produce `Results/English/{training_loss,validation_loss}.png` once a
  training run exists.
- Telugu: no loss curves (no training — see §2.2).

## 5. Generated speech — observations per test sentence

- **No test audios.** Training did not run by the 21:00 IST submission
  (§3), so no speech was synthesized. The five test sentences are in
  `English/test_sentences_en.txt` (E1 antidisestablishmentarianism;
  E2a/E2b conduct homograph noun vs verb; E3 rhythm; E4 "10:45 AM on
  12 November 2026" time/date expansion), and `inference.py` will
  produce `Results/English/test_generated_audios/<ID>.wav` once a
  checkpoint exists.
- Telugu: no test audios (no training — see §2.2).

## 6. Variance-adaptor analysis

- **No variance-adaptor plots.** Training did not run by the 21:00 IST
  submission (§3), so there are no duration/pitch/energy predictions to
  plot. `scripts/plot_variance.py` is in the repo and will produce
  `Results/English/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`
  once inference runs.
- Telugu: no variance plots (no training — see §2.2).

## 7. English vs Telugu — comparison, limitations, improvements

- English is the only trained model; Telugu could not be compared on
  audio quality because its alignment never completed (§2.2).
- Limitations (stated plainly): English has only 2 speaker embeddings
  (gender, not 117 speakers) because Svarah's parquet carries no
  per-speaker ID; 35/4154 utterances failed alignment and were dropped;
  **training did not run by the 21:00 IST submission (0 steps), so there
  is no trained model, no generated audio, and no loss curves in this
  submission** — the deliverable is the complete, verified pipeline
  (data selection, alignment, preprocessing, config, code, lexicon)
  plus this honest report.
- With more time/compute: more training steps; a real per-speaker ID for
  Svarah (parse speaker from audio paths); retry Telugu MFA alignment
  with `--num_jobs 1` (suspected multiprocessing stall at sat_3 iter 26)
  or on a machine with more RAM/CPU headroom; manual alignment fixes.

## 8. Reproducibility

- Environment: laptop (CPU-only, Intel Arc, no CUDA) did MFA alignment +
  preprocessing; training did not run by the 21:00 IST submission (Colab
  T4 started as a follow-on). PyTorch: CPU-only torch in the MFA env
  (python 3.11, montreal-forced-aligner 3.4.3).
- `requirements.txt` + system deps (espeak-ng, montreal-forced-aligner
  via conda/micromamba, build-essential for pyworld).
- Exact command sequence per language: see README.md runbook.
  Notable fixes applied on the laptop: `datasets` downgraded to 2.21.0
  (newer versions need torchcodec to decode audio); `setuptools`
  downgraded to 80.10.2 (84.x removed `pkg_resources`, which pyworld
  imports); three librosa calls changed to keyword form
  (`pad_center(..., size=...)`, `librosa.filters.mel(sr=..., n_fft=...,
  n_mels=..., fmin=..., fmax=...)`) because librosa ≥0.10 made those
  arguments keyword-only; `English/audio/stft.py` made device-aware
  (input's device, `.to(device)`) because the stock code called `.cuda()`
  unconditionally and the laptop torch is CPU-only.
- Config files: `English/config.yaml`, `Telugu/config.yaml`.
- Checkpoint → audio mapping: none — no checkpoint exists at submission
  (training not run, §3); vocoder weights: vendored HiFi-GAN universal
  (`hifigan/generator_universal.pth.tar`).
- GitHub repo URL: [TODO]. Submitted via Google Form
  (https://forms.gle/7tgHgKJLjpBXwcBj6) on 2026-11-07.
