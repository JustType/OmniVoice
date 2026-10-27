"""Cuts text streamed by an LLM into sentences that can be generated on their own.

A general sentence splitter breaks Slovak and Czech text at ordinals and abbreviations
("od 12. februára", "izba č. 5", "napr. raňajky"), and every piece would be its own
generation with a pause around it. So a sentence ends only where the next one visibly
starts: end punctuation, whitespace, then an uppercase letter. The word before the
punctuation must not be a number or a single letter ("12. Mája", "p. Novák").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

MIN_SENTENCE_CHARS = 20

_BOUNDARY = re.compile(r"[.!?…]+[\"'”’»)\]]*\s+")
_LEADING_PUNCTUATION = "\"'“„‘«(["


@dataclass(frozen=True)
class Segment:
    text: str
    # Offset in the stream's text just past this segment, so a client can resume after it.
    end: int


class Segmenter:
    """Accumulates a stream's text and returns its sentences once they are complete."""

    def __init__(self, min_chars: int = MIN_SENTENCE_CHARS) -> None:
        self._min_chars = min_chars
        self._text = ""
        self._cut = 0

    @property
    def length(self) -> int:
        return len(self._text)

    def push(self, text: str) -> list[Segment]:
        """Adds text; returns the sentences it completed. Sentences shorter than ``min_chars``
        are merged into the next one."""
        self._text += text
        segments: list[Segment] = []
        for match in _BOUNDARY.finditer(self._text, self._cut):
            following = match.end()
            if following >= len(self._text):
                break  # the next sentence has not started yet
            if not self._text[following].isupper() or _is_abbreviation(self._text, self._cut, match.start()):
                continue
            sentence = self._text[self._cut:following].strip()
            if len(sentence) < self._min_chars:
                continue
            segments.append(Segment(text=sentence, end=following))
            self._cut = following
        return segments

    def flush(self) -> list[Segment]:
        """The text not yet returned, as one segment, whatever its length."""
        rest = self._text[self._cut:].strip()
        self._cut = len(self._text)
        return [Segment(text=rest, end=self._cut)] if rest else []


def _is_abbreviation(text: str, start: int, punctuation: int) -> bool:
    words = text[start:punctuation].split()
    if not words:
        return False
    word = words[-1].lstrip(_LEADING_PUNCTUATION)
    return word.isdigit() or (len(word) == 1 and word.isalpha())
