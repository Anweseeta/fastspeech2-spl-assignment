#!/usr/bin/env python3
"""Build the Telugu G2P lexicon for MFA forced alignment (and phone-mode inference).

Two modes (must match model.telugu_g2p in Telugu/config.yaml):

  phone (default): word -> espeak-ng IPA phones via the `phonemizer` package
                   (backend='espeak', language='te'). Needs espeak-ng installed:
                       sudo apt-get install -y espeak-ng
                   Words the phonemizer fails on fall back to characters
                   (documented in the output log).

  char:            word -> its Telugu graphemes (community char-level trick:
                   no G2P needed; MFA aligns at grapheme level).

Output: Telugu/lexicon/telugu-lexicon.txt in MFA format:
    word<TAB>phone1 phone2 ...

Also writes telugu_phones.txt (observed phone inventory) and CHECKS every
phone against Telugu/text/symbols.py — any missing symbol is printed loudly,
because text_to_sequence silently drops unknown symbols (which would corrupt
training data without any error).

Usage (run from repo root):
    python scripts/build_te_lexicon.py --mode phone
    python scripts/build_te_lexicon.py --mode char
"""

import argparse
import os
import sys

LAB_GLOB_ROOT = os.path.join("Telugu", "raw_data", "IndicTTS_Telugu")
LEX_OUT = os.path.join("Telugu", "lexicon", "telugu-lexicon.txt")
INV_OUT = os.path.join("Telugu", "lexicon", "telugu_phones.txt")


def collect_words(lab_root):
    words = set()
    n_files = 0
    for dirpath, _, filenames in os.walk(lab_root):
        for fn in filenames:
            if not fn.endswith(".lab"):
                continue
            n_files += 1
            with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                for line in f:
                    words.update(line.strip().split())
    print("collected {} unique words from {} .lab files".format(len(words), n_files))
    return sorted(words)


def g2p_phone(words):
    """espeak-ng via phonemizer, one word at a time (clean word boundaries)."""
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    lex, failed = {}, []
    for w in words:
        try:
            out = phonemize(
                w,
                language="te",
                backend="espeak",
                separator=Separator(phone=" ", word="", syllable=""),
                strip=True,
                njobs=1,
            )
            phones = [p for p in out.strip().split(" ") if p]
            if not phones:
                raise ValueError("empty phonemization")
            lex[w] = phones
        except Exception as e:
            failed.append((w, str(e)))
            lex[w] = list(w)  # char-level fallback, documented
    return lex, failed


def g2p_char(words):
    return {w: list(w) for w in words}, []


def check_symbols(lex):
    """Verify every phone maps to a symbol in Telugu/text/symbols.py."""
    sys.path.insert(0, os.path.join("Telugu"))
    from text.symbols import symbols  # noqa: E402
    symset = set(symbols)
    phones = sorted({p for ph in lex.values() for p in ph})
    missing = [p for p in phones if "@" + p not in symset and p not in symset]
    with open(INV_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(phones) + "\n")
    print("phone inventory ({} phones) -> {}".format(len(phones), INV_OUT))
    if missing:
        print("!!! MISSING from text/symbols.py (add to _telugu_phones_ipa):")
        for p in missing:
            print("    " + p)
    else:
        print("all phones covered by text/symbols.py")
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["phone", "char"], default="phone")
    ap.add_argument("--lab_root", default=LAB_GLOB_ROOT)
    ap.add_argument("--out", default=LEX_OUT)
    args = ap.parse_args()

    words = collect_words(args.lab_root)
    if not words:
        raise SystemExit("no .lab files found under " + args.lab_root)

    if args.mode == "phone":
        lex, failed = g2p_phone(words)
    else:
        lex, failed = g2p_char(words)

    if failed:
        print("phonemizer failed on {} words (char fallback used):".format(len(failed)))
        for w, e in failed[:20]:
            print("   ", w, "->", e)

    # print a few samples so the student can eyeball espeak-ng quality
    print("samples:")
    for w in words[:5]:
        print("   ", w, "->", " ".join(lex[w]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for w in words:
            f.write(w + "\t" + " ".join(lex[w]) + "\n")
    print("wrote {} entries -> {}".format(len(lex), args.out))

    missing = check_symbols(lex)
    if missing:
        raise SystemExit("extend _telugu_phones_ipa in Telugu/text/symbols.py first")


if __name__ == "__main__":
    main()
