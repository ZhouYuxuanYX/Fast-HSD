"""INCLUDE multiple-choice answer extraction — ported from the legacy script.

Source: ``verification/src/INCLUDE/run_include_qwen2.5_simple.py`` (lines 140-163).
Three-pass regex with the same fallback behavior as the legacy script.
"""

from __future__ import annotations

import random
import re
from typing import Optional, Sequence

_RE_ANSWER_IS = re.compile(r"answer is \(?([A-D])\)?", re.IGNORECASE)
_RE_ANSWER_COLON = re.compile(r"[aA]nswer:\s*\(?([A-D])\)?")
_RE_LAST_LETTER = re.compile(r"\b([A-D])\b(?!.*\b[A-D]\b)", re.IGNORECASE | re.DOTALL)


def extract_letter(response: str) -> Optional[str]:
    """Return the predicted answer letter ``A``-``D`` or ``None``.

    Tries in order: ``answer is X`` (with optional parens), ``Answer: X``,
    last standalone ``A-D`` letter in the response.
    """
    m = _RE_ANSWER_IS.search(response)
    if m:
        return m.group(1).upper()
    m = _RE_ANSWER_COLON.search(response)
    if m:
        return m.group(1).upper()
    m = _RE_LAST_LETTER.search(response)
    if m:
        return m.group(1).upper()
    return None


def score_include(
    response: str,
    gold_letter: str,
    options: Optional[Sequence[str]] = None,
    rng: Optional[random.Random] = None,
) -> tuple:
    """Return ``(correct, predicted_letter)``.

    Mirrors legacy behaviour: if extraction fails, draw a uniform random
    letter (over ``options`` if given, else A-D) and count its correctness.
    ``rng`` may be supplied for deterministic random fallback.
    """
    gold = (gold_letter or "").strip().upper()
    pred = extract_letter(response)
    if pred is None:
        rng = rng or random
        n = len(options) if options else 4
        # Sample uniformly over the option indices, then map to A/B/C/D.
        pred = chr(ord("A") + rng.randrange(max(1, n)))
    return (pred == gold, pred)
