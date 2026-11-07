# Technical Report — section outline

Fill each section with your own observations. Every claim about training
should cite a figure/table in this repo (no invented numbers).

## 1. Implementation approach
- Reference: FastSpeech 2 (Ren et al., 2020, arXiv:2006.04558); codebase:
  ming024/FastSpeech2 (commit/branch + date cloned).
- Architecture components used as-is: Transformer encoder/decoder,
  variance adaptor (duration/pitch/energy predictors), PostNet, HiFi-GAN
  universal vocoder for waveform generation.
- Per-language pipeline: data selection → text normalization → forced
  alignment (MFA) → preprocessing (mel/pitch/energy/duration) → training →
  inference → variance-adaptor plots.
- Modifications vs stock: point to `docs/CHANGES.md` (summarize the big ones
  here: combined `config.yaml`, multi-speaker training, Telugu cleaner +
  symbol set, Telugu lexicon, `inference.py` variance-adaptor dumps).

## 2. Datasets
### 2.1 English — AI4Bharat Svarah
- Source, license, total size; why multi-speaker (117 speakers, ~5 min each —
  a single-speaker 5h split is impossible).
- Subsampling: per-speaker seeded shuffle (seed 42), fixed fraction to hit
  ~5h train / ~1h val, 5:1 split; cite `data_selection.md` (pool hours,
  achieved hours, per-speaker table).
- Preprocessing: 22050 Hz mono int16, `english_cleaners` on transcripts,
  MFA `english_us_arpa` alignment, phoneme-level pitch/energy with
  normalization (stats in `preprocessed_data/Svarah/stats.json`).
### 2.2 Telugu — IndicTTS Telugu
- Source, 2 speakers (male ~4.47h / female ~4.27h) — same honest
  multi-speaker note as above.
- Same selection method; cite `data_selection.md`.
- Preprocessing: Telugu text normalization (`scripts/text_normalize_te.py`:
  digits/times/ordinals → words), MFA `train_and_align` (no pretrained
  Telugu acoustic model exists) with the espeak-ng lexicon from
  `scripts/build_te_lexicon.py --mode phone` (or char mode — state which).
- State the final phone/grapheme inventory size.

## 3. Training configuration
- Table per language: batch 16, warm-up 4000, total steps configured
  (200k) vs steps actually run, save/val/synth cadence, optimizer settings,
  GPU type, wall-clock time, resume events (`--restore_step`).
- State explicitly if training stopped early and why (deadline) — the
  report must match the checkpoints in `*/checkpoints/`.

## 4. Loss-curve analysis
- Figures: `Results/English/training_loss.png`,
  `Results/English/validation_loss.png` (+ Telugu).
- Convergence behavior: when total loss plateaus; component losses
  (mel vs duration vs pitch/energy) — which converged slowest and why.
- Any instability (spikes, NaNs) and what was done; overfitting signs
  (train ↓ while val →/↑).

## 5. Generated speech — observations per test sentence
- Audio: `Results/<Lang>/test_generated_audios/<ID>.wav` (E1..E4, T1..T4).
- Per sentence: intelligibility, pronunciation of the hard bits
  (E1 antidisestablishmentarianism; E2a/E2b conduct homograph noun vs verb;
  E3 rhythm; E4 time/date expansion; T1 long compound; T2a/T2b పట్టు
  homograph; T3; T4 Telugu numerals/time), naturalness, artifacts
  (metallic/robotic, mispronunciations, skips/repeats).
- Which speaker ID was used for synthesis and why.

## 6. Variance-adaptor analysis
- Figures: `Results/<Lang>/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png`.
- Per test sentence: duration pattern (which phones are long/short and why —
  vowels vs consonants, phrase-final lengthening); pitch contour (declarative
  fall, stress on content words, question rise if any); energy (loudness on
  stressed syllables, dips at pauses).
- Note these are the adaptor's *predictions* at inference (denormalized to
  Hz), not ground truth.

## 7. English vs Telugu — comparison, limitations, improvements
- Which model sounds better and hypotheses why (data hours per speaker,
  alignment quality, G2P quality, phoneme vs grapheme).
- Failures: list them plainly (e.g. Telugu retroflexes, English stress
  errors, alignment rejects).
- Concrete improvements with more time/compute (more steps, pretrained
  English checkpoint fine-tune, better Telugu lexicon, manual alignment
  fixes).

## 8. Reproducibility
- Environment: OS, GPU, CUDA, PyTorch version; `requirements.txt` +
  system deps (espeak-ng, montreal-forced-aligner via conda, build-essential
  for pyworld).
- Exact command sequence per language (copy from README.md runbook).
- Config files: `English/config.yaml`, `Telugu/config.yaml`
  (multi_speaker, cleaner, g2p mode, acoustic params).
- Checkpoint → audio mapping: which `checkpoints/<step>.pth.tar` produced
  the submitted test audios; vocoder weights (vendored HiFi-GAN universal).
- GitHub repo URL + Google Form submission note.
