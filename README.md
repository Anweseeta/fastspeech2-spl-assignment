# FastSpeech 2 TTS — English & Telugu

## IIIT Hyderabad Speech Processing Lab — Evaluation Assignment

Implements **FastSpeech 2** (Ren et al., 2020) for text-to-speech in two languages:

- **English:** AI4Bharat Svarah
- **Telugu:** SPRINGLab IndicTTS_Telugu

Vocoder: vendored HiFi-GAN Universal.

**Repository:** https://github.com/Anweseeta/fastspeech2-spl-assignment  
**Technical report:** `docs/Technical_Report_DRAFT.md`  
**Candidate:** Anweseeta Sahoo

---

## Status (as of 2026-11-07)

| Stage | English | Telugu |
|---|---|---|
| Data selection | Done — 3462 train, 692 validation (5.01h / 1.02h) | Done — 845 train, 169 validation (4.88h / 1.11h) |
| Forced alignment (MFA) | Done — 4119 TextGrids out of 4154 | Blocked — see explanation below |
| Preprocessing | Done — 2 speakers, statistics computed | Not run — blocked on alignment |
| Training | Pending (GPU) | Pending |
| Test audio and plots | Pending | Pending |

### Telugu alignment limitation

Telugu alignment is blocked. There is no pretrained Telugu MFA acoustic model, so one had to be trained using `mfa train`.

Training hung at `sat_3`, iteration 26 of 35, across three attempts (Colab, laptop, and a resumed run on a cooled laptop). No model was saved and no TextGrids were exported.

Telugu is therefore submitted as pipeline-complete at the code and configuration level, with filelists and lexicon prepared. The alignment limitation is documented honestly in the technical report.

### Training status

Training had not started by the submission deadline (0 steps). English preprocessing finished shortly before the deadline, leaving insufficient time for a GPU training run.

Training and evaluation can continue after submission. Any subsequent results will be documented with their actual dates.

---

## Datasets

### English — `ai4bharat/Svarah` (gated)

- The available split is `test` (3 Parquet files, approximately 1.1 GB).
- Measured pool: 6656 utterances, 9.61 hours total.
- The Parquet data has no per-speaker ID. The `gender` field was therefore used as the speaker ID, producing two speaker categories (male/female), rather than 117 speakers.
- Selection: per-speaker seeded shuffle (seed 42), 5:1 train/validation split, with disjoint IDs.
- Alignment: `mfa align` using the pretrained `english_us_arpa` acoustic model and LibriSpeech lexicon (206,508 entries).
- 4119 TextGrids were exported out of 4154 utterances. The remaining 35 were dropped.

### Telugu — `SPRINGLab/IndicTTS_Telugu`

- Measured audio: 16 kHz mono, 36.71 hours across 2 speakers.
- Telugu uses speaker 0 (23.92 hours) with `multi_speaker: false`.
- Lexicon: 12,707 entries and 47 phones, generated using `scripts/build_te_lexicon.py --mode phone` with espeak-ng G2P.
- Text normalization: `scripts/text_normalize_te.py` (digits, times and ordinals converted into words).
- Telugu alignment remains blocked as described above.

---

## Pipeline

```text
Data selection
      ↓
Text normalization
      ↓
Forced alignment (MFA)
      ↓
Preprocessing (mel / pitch / energy / duration)
      ↓
Training
      ↓
Inference
      ↓
Variance-adaptor plots and evaluation
```

### Acoustic configuration

- Sampling rate: 22050 Hz
- Hop length: 256
- Mel channels: 80
- Maximum frequency: 8000 Hz

### Training configuration

- Batch size: 16
- Adam betas: [0.9, 0.98]
- Epsilon: 1e-9
- Gradient clipping: 1.0
- Warm-up: 4000 steps
- Checkpoint saving interval: 10000 steps

---

## Repository layout

```text
fastspeech2-spl-assignment/
│
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── test_sentences_en.txt
│
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── test_sentences_te.txt
│
├── scripts/
│   ├── make_filelists*.py
│   ├── text_normalize_en.py
│   ├── text_normalize_te.py
│   ├── build_te_lexicon.py
│   ├── plot_variance.py
│   └── plot_losses.py
│
├── hifigan/
│   └── generator_universal.pth.tar
│
├── Results/
│   ├── English/
│   └── Telugu/
│
├── docs/
│   ├── CHANGES.md
│   ├── Technical_Report_DRAFT.md
│   └── Technical_Report_Outline.md
│
├── data_selection.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

Raw audio, Parquet files and preprocessed features are not included in this repository because of their size. They are backed up separately. The filelists and `data_selection.md` record the selected utterances.

---

## Reproduction

### Install dependencies

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

sudo apt-get install -y espeak-ng build-essential
```

Install Montreal Forced Aligner using Conda:

```bash
conda install -c conda-forge montreal-forced-aligner
```

### Data selection

English requires a Hugging Face token because Svarah is gated.

```bash
export HF_TOKEN=<your-token>

python scripts/make_filelists_local_en.py

python scripts/make_filelists_local.py --lang te
```

### Forced alignment

English alignment:

```bash
mfa align English/raw_data/Svarah \
English/lexicon/librispeech-lexicon.txt \
english_us_arpa \
English/preprocessed_data/Svarah/TextGrid
```

Telugu alignment requires training a suitable acoustic model first. See the technical report for the documented limitation.

### Preprocessing

```bash
cd English
python preprocess.py --config config.yaml
cd ..
```

### Training (GPU)

```bash
cd English
python train.py --config config.yaml
cd ..
```

Resume training after interruption using:

```bash
python train.py --config config.yaml --restore_step <LAST_STEP>
```

### Inference

```bash
cd English

python inference.py --config config.yaml --restore_step <STEP> \
--speaker <speaker_name_from_speakers.json>

cd ..
```

### Plotting

```bash
python scripts/plot_losses.py --lang en

python scripts/plot_variance.py --lang en
```

Full command sequences and laptop-specific fixes (setuptools downgrade, librosa keyword-only calls and device-aware STFT for CPU-only PyTorch) are documented in `docs/CHANGES.md` and the technical report.

---

## Results

At submission:

- English: aligned and preprocessed.
- Telugu: pipeline code, filelists and lexicon prepared; alignment blocked.
- Training: not started (0 steps).
- Generated audio: pending.
- Loss curves: pending.

Test sentences are available in:

- `English/test_sentences_en.txt`
- `Telugu/test_sentences_te.txt`

Generated audios, loss plots and variance-adaptor plots will be placed in `Results/` as training and evaluation progress.

---

## Limitations

1. English has only two speaker embeddings based on gender because Svarah's Parquet data does not contain a per-speaker ID.
2. 35 out of 4154 English utterances failed alignment and were dropped.
3. Telugu MFA alignment did not complete after three attempts.
4. Telugu preprocessing and training could not start because alignment was blocked.
5. Training had not started by the submission deadline (0 steps). Therefore, speech quality claims await actual training and evaluation.

---

## Base implementation

FastSpeech 2 (Ren et al., 2020), paper: https://arxiv.org/abs/2006.04558

Original implementation: https://github.com/ming024/FastSpeech2

Modifications compared with the original implementation are listed in `docs/CHANGES.md`.
