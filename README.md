# FastSpeech 2 TTS — English & Telugu (IIIT-H SPL evaluation assignment)

Implements [FastSpeech 2](https://arxiv.org/abs/2006.04558) (from
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2)) separately for
**English** (AI4Bharat Svarah) and **Telugu** (IndicTTS Telugu), with test
synthesis, variance-adaptor analysis plots, and a technical report.

> **Honest-framing note (read first).** The assignment's "~5h train + ~1h val
> per language" implies single-speaker data, but neither dataset supports
> that: Svarah is ~9.6h across **117 speakers** (~5 min each) and IndicTTS
> Telugu is ~8.7h across **2 speakers**. Both models therefore train
> **multi-speaker with learned speaker embeddings** (`multi_speaker: true`),
> keeping the 5h/1h budget as the data-volume target. The selection
> (`data_selection.md`), configs, report and this README all state this
> explicitly. The full list of code changes vs the reference is in
> [`docs/CHANGES.md`](docs/CHANGES.md).

## Repo layout

```
English/                  # English model: train.py, preprocess.py, inference.py,
                          # config.yaml, test_sentences_en.txt, checkpoints/
Telugu/                   # Telugu model: same + Telugu text support
scripts/                  # make_filelists.py, text_normalize_{te,en}.py,
                          # build_te_lexicon.py, plot_variance.py, plot_losses.py
hifigan/                  # vendored HiFi-GAN UNIVERSAL vocoder weights
Results/{English,Telugu}/ # loss plots, test_generated_audios/, variance_adaptor_plots/
docs/                     # CHANGES.md, Technical_Report_Outline.md
data_selection.md         # filled by make_filelists.py (seed, hours, per-speaker table)
requirements.txt  .gitignore  .gitattributes (Git LFS for *.pth.tar)
```

Run every command from the **repo root** unless it says `cd English/` /
`cd Telugu/` (the language scripts expect to run from inside their dir).

## 0. Setup (Kaggle / Colab GPU)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
sudo apt-get install -y espeak-ng build-essential   # Telugu G2P + pyworld build
# Montreal Forced Aligner (needs conda):
conda install -c conda-forge montreal-forced-aligner
```

- HuggingFace: create an account, **accept the terms** on
  `ai4bharat/Svarah` (gated), create a read token → `export HF_TOKEN=...`
- **Kaggle**: enable GPU (P100), add `HF_TOKEN` as a secret, phone-verify the
  account. **Colab**: free T4, `!pip install` the same requirements.
- Strategy used for this deadline: **English on Kaggle P100, Telugu on Colab
  T4 in parallel** (~12h each for ~100–130k steps at batch 16 — the practical
  target; `total_step: 200000` in the configs is the ceiling).

### Session-wall survival
- Checkpoints save every 10k steps to `<Lang>/output/ckpt/<Ds>/`.
- **Kaggle**: after each session, "Save Version" / upload `output/ckpt` as a
  Kaggle dataset, or `cp -r` to `/kaggle/working` and persist. Resume with
  `python train.py --config config.yaml --restore_step <LAST_STEP>`.
- **Colab**: mount Drive and symlink: `ln -s /content/drive/MyDrive/fs2/ckpt
  English/output/ckpt`, same for `output/log`.
- Never let a disconnect lose more than one `save_step` of work.

## 1. Data — download + seeded selection (~5h train / ~1h val per language)

```bash
export HF_TOKEN=...   # needed for gated Svarah
python scripts/make_filelists.py --lang en --hf_token $HF_TOKEN
python scripts/make_filelists.py --lang te
```

This streams both HF datasets, resamples to 22050 Hz mono int16, cleans the
transcripts (English: `english_cleaners`; Telugu: `text_normalize_te.py`),
and writes `English/raw_data/Svarah/<speaker>/<uttid>.wav` + `.lab` (same for
Telugu). Selection: per-speaker seeded shuffle (seed 42), fixed fraction per
speaker to hit the hour targets, 5:1 train:val; train/val IDs are asserted
disjoint. Results + per-speaker table go to `data_selection.md`.

## 2. Forced alignment — phoneme durations (MFA)

The preprocessor needs `<preprocessed_path>/TextGrid/<speaker>/<uttid>.TextGrid`
with a tier named **phones**.

**English** (pretrained `english_us_arpa` acoustic model + dictionary):
```bash
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
# add OOV words from your .lab vocab first if MFA reports missing words:
#   mfa g2p --model english_us_arpa <oov.txt> >> <custom.dict>
mfa align English/raw_data/Svarah <your-english.dict> english_us_arpa \
    English/preprocessed_data/Svarah/TextGrid --clean --overwrite --num_jobs 4
```

**Telugu** (no pretrained MFA model exists → train one on your data):
```bash
python scripts/build_te_lexicon.py --mode phone   # espeak-ng G2P -> Telugu/lexicon/telugu-lexicon.txt
#   (--mode char for the grapheme-level fallback; must match model.telugu_g2p in Telugu/config.yaml)
mfa train_and_align Telugu/raw_data/IndicTTS_Telugu \
    Telugu/lexicon/telugu-lexicon.txt \
    Telugu/preprocessed_data/IndicTTS_Telugu/TextGrid --clean --overwrite --num_jobs 4
```
`build_te_lexicon.py` also verifies every phone exists in
`Telugu/text/symbols.py` (unknown symbols are silently dropped by
`text_to_sequence` — the script fails loudly instead).

## 3. Preprocess — mel / pitch / energy / duration features

```bash
cd English && python preprocess.py --config config.yaml && cd ..
cd Telugu  && python preprocess.py --config config.yaml && cd ..
```
Writes `preprocessed_data/<Ds>/{mel,pitch,energy,duration,train.txt,val.txt,
speakers.json,stats.json}`. Check `speakers.json` for the exact speaker names
(needed for inference `--speaker`).

## 4. Train

```bash
cd English && python train.py --config config.yaml && cd ..
cd Telugu  && python train.py --config config.yaml && cd ..
# resume after a disconnect:
cd English && python train.py --config config.yaml --restore_step 60000 && cd ..
```
Watch TensorBoard: `tensorboard --logdir English/output/log/Svarah`.
Practical target: **~100k–130k steps** for intelligible speech on this data
budget; stop early if the deadline forces it and say so in the report.

## 5. Synthesize the test sentences + dump variance-adaptor predictions

```bash
cd English && python inference.py --config config.yaml --restore_step 120000 \
    --speaker <speaker_name_from_speakers.json> && cd ..
cd Telugu && python inference.py --config config.yaml --restore_step 120000 \
    --speaker <speaker_name_from_speakers.json> [--g2p phone] && cd ..
```
- wavs → `Results/English/test_generated_audios/{E1,E2a,E2b,E3,E4}.wav`
  (Telugu: `T1..T4`)
- variance-adaptor dumps (per-phone duration / denormalized pitch Hz /
  energy + phone labels) → `<Lang>/output/va/<Ds>/{ID}.npz`
- Telugu `--g2p` must match `model.telugu_g2p` in `Telugu/config.yaml` and
  the lexicon mode, or train/inference symbol sequences won't line up.

## 6. Plots

```bash
python scripts/plot_losses.py --lang en && python scripts/plot_losses.py --lang te
# -> Results/{English,Telugu}/{training_loss.png,validation_loss.png}
python scripts/plot_variance.py --lang en && python scripts/plot_variance.py --lang te
# -> Results/{English,Telugu}/variance_adaptor_plots/{duration,pitch,energy}/{ID}.png
```

## 7. Technical report

Fill in [`docs/Technical_Report_Outline.md`](docs/Technical_Report_Outline.md)
(section-by-section, matching the assignment's required contents), export to
**`Technical_Report.pdf`** at the repo root. Every training claim must cite a
figure/table in this repo — no invented numbers.

## 8. Submit

1. `git lfs install`, commit everything (checkpoints via LFS), push to a
   **public** GitHub repo. If a checkpoint still exceeds limits, upload it to
   Drive and link it from here (the assignment allows an external link — keep
   it accessible during evaluation).
2. Submit the repo URL in the assignment's Google Form
   (https://forms.gle/7tgHgKJLjpBXwcBj6) before **2026-11-07 00:00**.

## Troubleshooting
- `espeak-ng` missing → `build_te_lexicon.py --mode phone` and Telugu
  phone-mode inference fall back per-word (see script output); install it.
- MFA OOV words → extend the dictionary, re-run with `--overwrite`.
- Empty phone sequence at inference → a phone is missing from
  `Telugu/text/symbols.py::_telugu_phones_ipa`; add it (see `docs/CHANGES.md`).
- CUDA OOM → lower `optimizer.batch_size` in `config.yaml` (16 fits a
  P100/T4 at these dims).
- `phonemizer` Separator API differs by version → `build_te_lexicon.py`
  prints samples; verify phones look like IPA before the full run.
