"""SPL assignment: test-sentence inference for the Telugu (IndicTTS) model.

Adapted from the stock synthesize.py, with Telugu support added:
  - reads a single combined config.yaml (--config)
  - reads test sentences from a file of "ID|text" lines
  - Telugu G2P modes (--g2p):
      * phone: espeak-ng IPA phones via lexicon/telugu-lexicon.txt
               (OOV words -> phonemizer on the fly -> char fallback)
      * char : raw Telugu graphemes, no G2P (must match training mode)
  - synthesizes with a chosen multi-speaker speaker ID
  - saves 22050 Hz wavs to Results/Telugu/test_generated_audios/
  - ALSO dumps the variance adaptor's per-phone predictions
    (duration / pitch / energy) for scripts/plot_variance.py

The --g2p mode MUST match model.telugu_g2p in config.yaml and the mode used
in scripts/build_te_lexicon.py, otherwise train/inference symbol sequences
will not line up.

Run from inside Telugu/:
    python inference.py --config config.yaml --restore_step 100000 \\
        --speaker <speaker_name_from_speakers.json> [--g2p phone]
"""

import argparse
import json
import os
import re

import numpy as np
import torch
import yaml
from scipy.io import wavfile

from text import text_to_sequence
from text.symbols import symbols
from utils.model import get_model, get_vocoder, vocoder_infer
from utils.tools import to_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


def read_te_lexicon(lex_path):
    """MFA-format lexicon: 'word<TAB>phone1 phone2 ...' -> {word: [phones]}."""
    lexicon = {}
    with open(lex_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) >= 2:
                lexicon[parts[0]] = parts[1:]
    return lexicon


def _phonemize_word_espeak(word):
    """Fallback G2P for OOV words via phonemizer + espeak-ng (language 'te').

    Returns a list of IPA phone strings, or None if phonemizer/espeak-ng is
    not available (caller falls back to characters).
    """
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
    except ImportError:
        return None
    try:
        out = phonemize(
            word,
            language="te",
            backend="espeak",
            separator=Separator(phone=" ", word="", syllable=""),
            strip=True,
            njobs=1,
        )
        phones = [p for p in out.strip().split(" ") if p]
        return phones or None
    except Exception as e:  # e.g. espeak-ng missing
        print("  [warn] espeak-ng phonemizer failed for '{}': {}".format(word, e))
        return None


def preprocess_telugu(text, preprocess_config, lexicon, g2p_mode):
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    if g2p_mode == "char":
        # Character-level: Telugu graphemes map directly to symbols
        # (telugu_cleaners preserves Unicode; no braces needed).
        print("Raw Text Sequence: {}".format(text))
        sequence = np.array(text_to_sequence(text, cleaners))
        return np.array(sequence)

    # Phone-level: words -> IPA phones via lexicon, OOV via espeak-ng.
    phones = []
    for w in text.split():
        if w in lexicon:
            phones += lexicon[w]
        else:
            oov_phones = _phonemize_word_espeak(w)
            if oov_phones is not None:
                print("  [oov] '{}' -> {}".format(w, " ".join(oov_phones)))
                phones += oov_phones
            else:
                print("  [oov] '{}' -> char fallback".format(w))
                phones += list(w)  # chars are in the symbol table too
        phones.append("sp")  # word boundary
    if phones and phones[-1] == "sp":
        phones = phones[:-1]

    phones_braced = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones_braced))
    sequence = np.array(text_to_sequence(phones_braced, cleaners))
    if len(sequence) == 0:
        raise ValueError(
            "Empty phone sequence for '{}' — a phone is probably missing "
            "from text/symbols.py (extend _telugu_phones_ipa).".format(text))
    return np.array(sequence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--test_file", type=str, default="test_sentences_te.txt",
                        help="file of 'ID|text' lines")
    parser.add_argument("--speaker", type=str, required=True,
                        help="speaker NAME as in "
                             "preprocessed_data/IndicTTS_Telugu/speakers.json")
    parser.add_argument("--g2p", type=str, choices=["phone", "char"], default=None,
                        help="Telugu G2P mode; defaults to model.telugu_g2p in config")
    parser.add_argument("--audio_out", type=str,
                        default="../../Results/Telugu/test_generated_audios")
    parser.add_argument("--va_out", type=str, default="./output/va/IndicTTS_Telugu",
                        help="where to dump variance-adaptor prediction .npz files")
    args = parser.parse_args()

    full_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocess_config = full_config["preprocess"]
    model_config = full_config["model"]
    train_config = full_config["train"]
    configs = (preprocess_config, model_config, train_config)

    g2p_mode = args.g2p or model_config.get("telugu_g2p", "phone")
    print("Telugu G2P mode: {}".format(g2p_mode))

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"],
                           "speakers.json")) as f:
        speaker_map = json.load(f)
    if args.speaker not in speaker_map:
        raise ValueError("Unknown speaker '{}'. Available: {}".format(
            args.speaker, sorted(speaker_map)))
    speaker_id = speaker_map[args.speaker]
    print("Synthesizing with speaker '{}' (id {})".format(args.speaker, speaker_id))

    with open(os.path.join(preprocess_config["path"]["preprocessed_path"],
                           "stats.json")) as f:
        stats = json.load(f)
    pitch_mean, pitch_std = stats["pitch"][2], stats["pitch"][3]
    energy_mean, energy_std = stats["energy"][2], stats["energy"][3]

    lexicon = {}
    if g2p_mode == "phone":
        lexicon = read_te_lexicon(preprocess_config["path"]["lexicon_path"])
        print("Loaded {} lexicon entries".format(len(lexicon)))

    model = get_model(args, configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    os.makedirs(args.audio_out, exist_ok=True)
    os.makedirs(args.va_out, exist_ok=True)

    with open(args.test_file, encoding="utf-8") as f:
        tests = [line.rstrip("\n").split("|", 1) for line in f if line.strip()]

    for utt_id, text in tests:
        print("=== {} ===".format(utt_id))
        sequence = preprocess_telugu(text, preprocess_config, lexicon, g2p_mode)

        ids = [utt_id]
        raw_texts = [text]
        speakers = np.array([speaker_id])
        texts = np.array([sequence])
        text_lens = np.array([len(sequence)])
        batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
        batch = to_device(batch, device)

        with torch.no_grad():
            # predictions: (mel, mel_postnet, p_pred, e_pred, log_d_pred,
            #               d_rounded, src_masks, mel_masks, src_lens, mel_lens)
            output = model(*(batch[2:]))

        src_len = output[8][0].item()
        mel_len = output[9][0].item()

        # --- variance adaptor predictions (per phone, phoneme-level) ---
        duration_frames = output[5][0, :src_len].detach().cpu().numpy().astype(int)
        pitch_norm = output[2][0, :src_len].detach().cpu().numpy()
        energy_norm = output[3][0, :src_len].detach().cpu().numpy()
        pitch_hz = pitch_norm * pitch_std + pitch_mean      # denormalize
        energy = energy_norm * energy_std + energy_mean     # denormalize
        duration_sec = duration_frames * hop_length / sampling_rate

        np.savez(
            os.path.join(args.va_out, "{}.npz".format(utt_id)),
            duration_frames=duration_frames,
            duration_sec=duration_sec,
            pitch_hz=pitch_hz,
            energy=energy,
            pitch_norm=pitch_norm,
            energy_norm=energy_norm,
        )
        with open(os.path.join(args.va_out, "{}_phones.json".format(utt_id)),
                  "w", encoding="utf-8") as f:
            labels = [s[1:] if s.startswith("@") else s
                      for s in (_id_to_symbol[i] for i in sequence)]
            json.dump(labels, f, ensure_ascii=False)

        # --- waveform via HiFi-GAN universal vocoder ---
        mel = output[1][0, :mel_len].detach().transpose(0, 1).unsqueeze(0)
        wavs = vocoder_infer(mel, vocoder, model_config, preprocess_config,
                             lengths=np.array([mel_len * hop_length]))
        wav_path = os.path.join(args.audio_out, "{}.wav".format(utt_id))
        wavfile.write(wav_path, sampling_rate, wavs[0])
        print("wrote {}".format(wav_path))

    print("Done. {} utterances.".format(len(tests)))


if __name__ == "__main__":
    main()
