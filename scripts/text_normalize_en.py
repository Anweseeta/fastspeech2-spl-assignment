#!/usr/bin/env python3
"""English text normalization notes + light wrapper for the SPL assignment.

The heavy lifting for English is done by `english_cleaners`
(English/text/cleaners.py): unidecode -> lowercase -> number expansion ->
abbreviation expansion -> whitespace collapse. That cleaner runs automatically
inside the preprocessor (on .lab text) and inside inference.py, so most text
needs NO pre-processing.

This module exists for two edge cases:
  1. Tokens english_cleaners cannot expand (URLs, email-like IDs, mixed
     alphanumeric codes such as "req200030795"): expand_numbers handles pure
     digit runs; anything else should be rewritten by hand before training.
  2. The graded test sentences: E2a/E2b grade the conduct/conduct homograph
     (noun CON-duct vs verb con-DUCT) and E4 grades "10:45 AM" / "12 November
     2026" expansion. english_cleaners + the g2p_en fallback in inference.py
     handle these; this script lets you verify the expansion offline.

Usage:
    python text_normalize_en.py "The meeting will start at 10:45 AM on 12 November 2026"
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "English"))

from text.cleaners import english_cleaners  # noqa: E402


def normalize_en_text(text):
    """Apply the exact cleaner used in training/inference. Import-safe."""
    return english_cleaners(text)


if __name__ == "__main__":
    demos = sys.argv[1:] or [
        "The meeting will start at 10:45 AM on 12 November 2026, so everyone should arrive on time.",
        "The orchestra's conduct during the ceremony was praised by the audience.",
        "The researchers will conduct an experiment to evaluate the proposed method.",
    ]
    for d in demos:
        print("IN :", d)
        print("OUT:", normalize_en_text(d))
        print()
