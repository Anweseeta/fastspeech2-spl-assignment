""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

CHANGED (SPL assignment, Telugu support): the stock symbol set only covers
ASCII + ARPAbet (+ pinyin, unused here). Telugu adds (1) Telugu Unicode
graphemes for character-level modeling and (2) the espeak-ng Telugu phone
inventory (IPA, "@"-prefixed like ARPAbet) for phone-level modeling.

How mapping works (see text/__init__.py): phone strings wrapped in curly
braces, e.g. "{k aː}", have each token "@"-prefixed and looked up in this
table. Any token missing here is SILENTLY DROPPED by _should_keep_symbol, so
scripts/build_te_lexicon.py verifies full coverage of the lexicon against
this table before training.
"""

from text import cmudict

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]

# --- ADDED: Telugu graphemes (U+0C00-0C7F block) for character-level modeling.
# Vowels/diacritics/consonants/vowel signs/virama/digits. Used when Telugu is
# modeled at the grapheme level (no G2P required); each char maps directly.
_telugu_graphemes = (
    [chr(cp) for cp in range(0x0C01, 0x0C04)]  # signs (anusvara, visarga, ...)
    + [chr(cp) for cp in range(0x0C05, 0x0C15)]  # independent vowels అ-ఔ
    + [chr(cp) for cp in range(0x0C15, 0x0C3A)]  # consonants క-హ
    + [chr(cp) for cp in range(0x0C3E, 0x0C4E)]  # vowel signs + virama ్
    + [chr(cp) for cp in range(0x0C66, 0x0C70)]  # digits ౦-౯
)

# --- ADDED: espeak-ng Telugu phone inventory (IPA), "@"-prefixed.
# Produced by scripts/build_te_lexicon.py --mode phone (phonemizer, espeak
# backend, language 'te'). If the phonemizer emits a symbol not listed here,
# add it: missing symbols are silently dropped at text_to_sequence time.
_telugu_phones_ipa = [
    # stops / affricates
    "p", "pʰ", "b", "bʱ",
    "t̪", "t̪ʰ", "d̪", "d̪ʱ",
    "ʈ", "ʈʰ", "ɖ", "ɖʱ",
    "tʃ", "tʃʰ", "dʒ", "dʒʱ",
    "k", "kʰ", "ɡ", "ɡʱ",
    # espeak-ng Telugu variants actually observed in the lexicon
    # (ʰ U+02B0 aspiration; palatals as c/ɟ rather than tʃ/dʒ)
    "bʰ", "dʰ", "tʰ", "ɖʰ", "ɡʰ", "cʰ", "ɟ", "ɟʰ",
    # nasals
    "m", "n̪", "n", "ɳ", "ɲ", "ŋ",
    # liquids / glides
    "ɾ", "r", "l", "ɭ", "ʋ", "j",
    # palatalization modifier (observed in espeak-ng output)
    "ʲ",
    # fricatives
    "s", "ʂ", "ɕ", "h", "ɦ",
    # vowels
    "a", "aː", "i", "iː", "u", "uː",
    "e", "eː", "o", "oː", "ai", "au", "ə", "r̥",
    # IPA diphthongs as emitted by espeak-ng
    "aɪ", "aʊ",
]
_telugu_phones = ["@" + p for p in _telugu_phones_ipa]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _telugu_graphemes
    + _telugu_phones
    + _silences
)
