#!/usr/bin/env python3
"""Telugu text normalization for the SPL assignment.

Expands Telugu-script digits, Arabic numerals, clock times and simple
ordinals into Telugu words BEFORE forced alignment / training, so the
acoustic model never has to learn digit pronunciation from sparse data.

Test sentence this must handle (T4):
  "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, ..."

Usage:
    from text_normalize_te import normalize_te_text
    normalize_te_text("10:45")  # -> "పది గంటల నలభై ఐదు నిమిషాలు"
"""

import re

# Telugu-script digits -> words
TE_DIGIT_WORDS = {
    "౦": "సున్నా", "౧": "ఒకటి", "౨": "రెండు", "౩": "మూడు", "౪": "నాలుగు",
    "౫": "ఐదు", "౬": "ఆరు", "౭": "ఏడు", "౮": "ఎనిమిది", "౯": "తొమ్మిది",
}

_ONES = ["", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు",
         "ఎనిమిది", "తొమ్మిది"]
_TEENS = ["పది", "పదకొండు", "పన్నెండు", "పదమూడు", "పద్నాలుగు", "పదిహేను",
          "పదహారు", "పదిహేడు", "పద్దెనిమిది", "పంతొమ్మిది"]
_TENS = ["", "", "ఇరవై", "ముప్పై", "నలభై", "యాభై", "అరవై", "డెబ్భై",
         "ఎనభై", "తొంభై"]


def _two_digit(n):
    """0..99 -> Telugu words."""
    if n < 10:
        return "సున్నా" if n == 0 else _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + ("" if ones == 0 else " " + _ONES[ones])


def num_to_te_words(n):
    """0..99999 -> Telugu words (supports years like 2026)."""
    if n < 0 or n > 99999:
        # fall back to digit-by-digit for out-of-range numbers
        return " ".join(_ONES[int(d)] if d != "0" else "సున్నా" for d in str(abs(n)))
    if n < 100:
        return _two_digit(n)
    if n < 1000:
        h, rest = divmod(n, 100)
        # 100 = వంద, 200 = రెండు వందల, 300 = మూడు వందల ...
        head = "వంద" if h == 1 else _ONES[h] + " వందల"
        return head if rest == 0 else head + " " + _two_digit(rest)
    th, rest = divmod(n, 1000)
    # 1000 = వెయ్యి, 2000 = రెండు వేల, 2026 = రెండు వేల ఇరవై ఆరు
    head = "వెయ్యి" if th == 1 else _two_digit(th) + " వేల"
    if rest == 0:
        return head
    return head + " " + (num_to_te_words(rest) if rest >= 100 else _two_digit(rest))


def ordinal_te(n):
    """n -> Telugu ordinal word, e.g. 12 -> పన్నెండవ (for '12వ తేదీ')."""
    w = num_to_te_words(n)
    # take the last word and attach the ordinal suffix
    *head, last = w.split(" ")
    if last.endswith("ు"):
        last = last[:-1] + "వ"
    else:
        last = last + "వ"
    return " ".join(head + [last])


def normalize_te_text(text):
    """Full normalization pipeline for a Telugu utterance."""
    # 1. clock times "10:45" / "10.45" -> "పది గంటల నలభై ఐదు నిమిషాలు".
    #    A trailing గంటలకు/గంటల/గంటకు is folded in ("10:45 గంటలకు" ->
    #    "పది గంటల నలభై ఐదు నిమిషాలకు"), not duplicated.
    def _time(m):
        hh, mm = int(m.group(1)), int(m.group(2))
        has_case = m.group(3)
        out = num_to_te_words(hh) + " గంటల"
        if mm:
            out += " " + num_to_te_words(mm) + " నిమిషాలు"
        if has_case:
            if mm:
                out = out[: -len("లు")] + "లకు"  # నిమిషాలు -> నిమిషాలకు
            else:
                out = out + "కు"  # గంటల -> గంటలకు
        return out
    text = re.sub(r"(\d{1,2})[:.](\d{2})\s*(గంటలకు|గంటల|గంటకు)?", _time, text)

    # 2. ordinals like "12వ" -> "పన్నెండవ"
    text = re.sub(r"\b(\d+)\s*వ\b", lambda m: ordinal_te(int(m.group(1))), text)

    # 3. remaining Arabic digit runs -> words
    text = re.sub(r"\b\d+\b", lambda m: num_to_te_words(int(m.group(0))), text)

    # 4. Telugu-script digits -> words
    text = "".join(TE_DIGIT_WORDS.get(ch, ch) + (" " if ch in TE_DIGIT_WORDS else "")
                   for ch in text)

    # 5. punctuation / whitespace cleanup (keep , . ? ! ; : -)
    text = text.replace("।", ".").replace("॥", ".")
    text = re.sub(r"[“”\"']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    demos = [
        "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉండండి.",
        "ధర ౧౨౩ రూపాయలు.",
    ]
    for d in demos:
        print("IN :", d)
        print("OUT:", normalize_te_text(d))
        print()
