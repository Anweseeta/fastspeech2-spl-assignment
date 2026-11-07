# Changes vs stock ming024/FastSpeech2

Reference: https://github.com/ming024/FastSpeech2 (cloned 2026-11-03).
Each language directory (`English/`, `Telugu/`) is a full copy of the
reference with the adaptations below. `scripts/` replaces the stock
`prepare_align.py` dataset-specific staging (see "removed").

## Both languages
- `train.py`: accepts a single `--config config.yaml` (combined
  `preprocess:` / `model:` / `train:` sections) instead of three separate
  yaml files. Training loop itself is stock.
- `preprocess.py`: same single-`--config` change; runs the stock
  `Preprocessor` on the combined config's `preprocess:` section.
- `config.yaml` (new): the three stock yamls merged into one file.
  Training config: `batch_size: 16`, `total_step: 200000` (practical free-GPU
  target ~100k–130k steps — see README), `save_step: 10000`,
  `warm_up_step: 4000`. Acoustic config fixed to the HiFi-GAN universal
  requirements: 22050 Hz, hop 256, 80 mels, fmax 8000.
- `utils/model.py::get_vocoder`: HiFi-GAN weights path comes from
  `model.vocoder.weights_path` (falls back to the repo-root
  `hifigan/generator_universal.pth.tar`); `config.json` is resolved relative
  to the imported `hifigan` package instead of the CWD. `torch.load` uses
  `map_location=device`.
- `inference.py` (new, from `synthesize.py`): batch synthesis over a
  `test_sentences_*.txt` file (`ID|text` lines) with an explicit `--speaker`
  name; saves 22050 Hz wavs AND dumps per-utterance variance-adaptor
  predictions (`duration_frames`, denormalized `pitch_hz`/`energy`, phone
  labels) as `.npz` + `_phones.json` for `scripts/plot_variance.py`.
- `hifigan/`: code + `config.json` only; the 55 MB
  `generator_universal.pth.tar` weights live once at the repo-root `hifigan/`
  (vendored from the reference repo, unzipped).
- `multi_speaker: true` for both models (see honest-deviation note below).
- Removed: `prepare_align.py`, `preprocessor/ljspeech.py`,
  `preprocessor/aishell3.py`, `preprocessor/libritts.py` — dataset staging is
  now `scripts/make_filelists.py` (HF download + resample + seeded split).

## English (`English/`) only
- `text/symbols.py`: stock ARPAbet set; removed the unused `pinyin` import
  (Mandarin not used — drops the pypinyin dependency).
- `text/cleaners.py`, `text/__init__.py`: stock (`english_cleaners`, which
  expands numbers/abbreviations — this is what handles E4's "10:45 AM /
  12 November 2026").
- `lexicon/librispeech-lexicon.txt`: copied from the reference for the
  inference-time G2P lexicon lookup.
- `inference.py` uses the stock `preprocess_english` (lexicon + `g2p_en`).

## Telugu (`Telugu/`) only — the main per-language work
- `text/cleaners.py`: added `telugu_cleaners` — whitespace normalization
  ONLY. No `unidecode`, no lowercasing: the stock `english_cleaners` would
  destroy Telugu script. Number/time/ordinal expansion happens beforehand in
  `scripts/text_normalize_te.py`.
- `text/symbols.py`: added (1) Telugu Unicode graphemes
  (U+0C01–0C03, U+0C05–0C14, U+0C15–0C39, U+0C3E–0C4D, U+0C66–0C6F) for
  character-level modeling, and (2) the espeak-ng Telugu IPA phone inventory,
  "@"-prefixed like ARPAbet so `{...}`-wrapped phone strings map through
  `text_to_sequence`'s brace path. Dropped the unused `pinyin` import.
  Any phone missing here is SILENTLY dropped at mapping time —
  `scripts/build_te_lexicon.py` verifies coverage and fails loudly.
- `config.yaml`: `text_cleaners: ["telugu_cleaners"]`, `language: "te"`,
  `model.telugu_g2p: "phone"` (must match the lexicon mode and inference
  `--g2p`), Telugu lexicon path.
- `inference.py`: `preprocess_telugu` with `--g2p phone|char`. Phone mode:
  word → IPA phones via `lexicon/telugu-lexicon.txt`, OOV → espeak-ng
  `phonemizer` on the fly → character fallback. Char mode: raw graphemes.
- `scripts/build_te_lexicon.py` (new): espeak-ng/`phonemizer` Telugu G2P →
  MFA-format lexicon, with char-level fallback and symbol-coverage check.
- `scripts/text_normalize_te.py` (new): Telugu digits, Arabic numerals,
  clock times ("10:45 గంటలకు" → "పది గంటల నలభై ఐదు నిమిషాలకు"), ordinals
  ("12వ" → "పన్నెండవ") → words.

## Test-sentence transcription fixes (OCR artifacts in the assignment PDF)
- E4: "andaroo" → "everyone" (PDF OCR artifact; sentence is "…so everyone
  should arrive on time.").
- E3: "rhyth m" → "rhythm". Telugu T1–T4: removed spurious intra-word spaces
  from PDF text extraction (e.g. "విశ్లే షించారు" → "విశ్లేషించారు").
  Originals preserved in the PDF; cleaned forms are what's synthesized.

## Honest-deviation note (also in data_selection.md + README + report)
The assignment's "~5h train + ~1h val per language" reads as single-speaker,
but neither dataset allows it: Svarah is ~9.6h over 117 speakers (~5 min
each); IndicTTS Telugu is ~8.7h over 2 speakers. Both models therefore train
multi-speaker with learned speaker embeddings, keeping the 5h/1h budget as
the data-volume target. The selection script, configs, report outline and
README all state this explicitly instead of pretending otherwise.
