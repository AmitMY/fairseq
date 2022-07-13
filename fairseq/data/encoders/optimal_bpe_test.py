import os
import tempfile
import unittest

from .optimal_bpe import OptimalBPE, OptimalBPETokenizer, OptimalBPEConfig


class OptimalBPETokenizerTestCase(unittest.TestCase):
    def test_semi_markov_segment_bs_brute_segment(self):
        vocabulary = {"a": 5, "b": 9, "c": 50, "ab": 11, "bc": 20, "cb": 31}
        tokenizer = OptimalBPETokenizer(vocabulary, '-')
        word = "abccbaba"

        segmentation, min_len1, max_freq1 = tokenizer.semi_markov_segment(word)
        min_len2, max_freq2 = tokenizer.brute_segment(word)

        self.assertEqual(min_len1, min_len2)
        self.assertEqual(max_freq1, max_freq2)

    def test_semi_markov_every_segment_in_vocab(self):
        vocabulary = {"a": 5, "b": 9, "c": 50, "ab": 11, "bc": 20, "cb": 31}
        tokenizer = OptimalBPETokenizer(vocabulary, '-')
        word = "abccbaba"

        segmentation, _, _ = tokenizer.semi_markov_segment(word)
        for s in segmentation:
            self.assertTrue(s in vocabulary)

    def test_semi_markov_segment_tokens_replicate_source(self):
        vocabulary = {"a": 5, "b": 9, "c": 50, "ab": 11, "bc": 20, "cb": 31}
        tokenizer = OptimalBPETokenizer(vocabulary, '-')
        word = "abccbaba"

        segmentation, _, _ = tokenizer.semi_markov_segment(word)

        self.assertEqual("".join(segmentation), word)

    def test_word_segmentation_cache_added(self):
        vocabulary = {"a": 5, "b": 9, "c": 50, "ab": 11, "bc": 20, "cb": 31}
        tokenizer = OptimalBPETokenizer(vocabulary, '-')
        word = "abccbaba"
        expected = ['ab', 'c', 'cb', 'ab', 'a']

        segments = tokenizer.segment_word(word)
        self.assertEqual(segments, expected)
        self.assertEqual(tokenizer.cache, {word: expected})

    def test_segment_words_multiple_words(self):
        vocabulary = {"a ": 5, "a": 5, "b": 9, "c": 50, "ab ": 11,"ab": 11, "bc": 20, "cb": 31}
        tokenizer = OptimalBPETokenizer(vocabulary, '-')
        words = ["abccbaba", "bcaab"]
        tokens = list(tokenizer.segment_words(words))
        self.assertEqual(['ab-', 'c-', 'cb-', 'ab-', 'a', 'bc-', 'a-', 'ab'], tokens)

    def test_sentence_segmentation(self):
        vocabulary = {"a": 5, "b": 9, "c": 50, "ab": 11, "bc": 20, "cb": 31}
        tokenizer = OptimalBPETokenizer(vocabulary, '-')
        sentence = "abccbaba bcaab\n"
        tokenized = tokenizer.segment(sentence)
        self.assertEqual("ab- c- cb- ab- a bc- a- ab", tokenized)


# def test_semi_markov_segment_unknown_tokens(self):
#     vocabulary = {"a": 5, "a✡": 1, "b": 9, "c": 50, "ab": 11, "bc": 20, "cb": 31}
#     tokenizer = OptimalBPETokenizer(vocabulary, '-')
#     sentence = "abccbaba def"
#     tokenized = tokenizer.segment(sentence)
#     self.assertEqual(tokenized, "ab- c- cb- ab- a✡ bc- a- ab✡")

class OptimalBPETestCase(unittest.TestCase):
    def test_sentence_segmentation(self):
        fd, path = tempfile.mkstemp(suffix=".codes")
        try:
            with open(path, 'w') as tmp:
                tmp.writelines([
                    '#version: 0.2\n',
                    'a ✡\n',
                    'a b\n',
                    'b c\n',
                    'c b\n',
                    'a b✡\n',
                ])
            cfg = OptimalBPEConfig(bpe_codes=path)
            encoder = OptimalBPE(cfg)
        finally:
            os.remove(path)

        sentence = "abccbaba bcaab\n"
        tokenized = encoder.encode(sentence)
        self.assertEqual(tokenized, "ab@@ c@@ cb@@ ab@@ a bc@@ a@@ ab")


if __name__ == '__main__':
    unittest.main()
