import unittest

import support  # noqa: F401

from server.segmenter import Segmenter


def _segments(text: str, chunk: int = 4) -> list[tuple[str, int]]:
    """Feeds ``text`` the way an LLM streams it, then ends it."""
    segmenter = Segmenter()
    segments = []
    for i in range(0, len(text), chunk):
        segments += segmenter.push(text[i:i + chunk])
    segments += segmenter.flush()
    return [(segment.text, segment.end) for segment in segments]


class SegmenterTests(unittest.TestCase):
    def test_ordinals_and_abbreviations_do_not_end_a_sentence(self):
        text = "Máte rezerváciu od 12. 2. do 15. 2. 2027, izba č. 5, napr. raňajky sú od 7:00. Cena je 120,50 €."
        self.assertEqual(
            [s for s, _ in _segments(text)],
            [
                "Máte rezerváciu od 12. 2. do 15. 2. 2027, izba č. 5, napr. raňajky sú od 7:00.",
                "Cena je 120,50 €.",
            ],
        )

    def test_a_capitalised_word_after_a_number_or_initial_stays_in_the_sentence(self):
        text = "Pán p. Novák príde 12. Mája o druhej. Ďakujem za trpezlivosť."
        self.assertEqual(
            [s for s, _ in _segments(text)],
            ["Pán p. Novák príde 12. Mája o druhej.", "Ďakujem za trpezlivosť."],
        )

    def test_sentences_end_at_question_and_exclamation_marks(self):
        text = "Jasné, k rezervácii Vám rado poviem všetko! Čo by ste sa chceli opýtať? Ste tam?"
        self.assertEqual(
            [s for s, _ in _segments(text)],
            ["Jasné, k rezervácii Vám rado poviem všetko!", "Čo by ste sa chceli opýtať?", "Ste tam?"],
        )

    def test_a_short_sentence_is_merged_into_the_next(self):
        self.assertEqual(
            [s for s, _ in _segments("Dobrý deň! Tu je Nina z hotela Ambiente.")],
            ["Dobrý deň! Tu je Nina z hotela Ambiente."],
        )

    def test_a_sentence_is_returned_only_once_the_next_one_starts(self):
        segmenter = Segmenter()
        self.assertEqual(segmenter.push("Tu je Nina z hotela Ambiente. "), [])
        self.assertEqual([s.text for s in segmenter.push("Ako")], ["Tu je Nina z hotela Ambiente."])

    def test_offsets_point_past_each_segment(self):
        text = "Tu je Nina z hotela Ambiente. Ako Vám môžem pomôcť?"
        segments = _segments(text)
        self.assertEqual([end for _, end in segments], [text.index("Ako"), len(text)])
        self.assertEqual(text[segments[0][1]:].strip(), segments[1][0])

    def test_flush_returns_nothing_for_blank_text(self):
        segmenter = Segmenter()
        segmenter.push("   ")
        self.assertEqual(segmenter.flush(), [])


if __name__ == "__main__":
    unittest.main()
