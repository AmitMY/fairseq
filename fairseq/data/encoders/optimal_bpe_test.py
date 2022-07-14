import os
import tempfile
import unittest

from .optimal_bpe import OptimalBPE, OptimalBPEConfig


class OptimalBPETestCase(unittest.TestCase):
    def test_sentence_segmentation(self):
        fd, path = tempfile.mkstemp(suffix=".codes")
        try:
            with open(path, 'w') as tmp:
                tmp.writelines([
                    '#version: 0.2\n',
                    'a b\n',
                    'b c\n',
                    'c b\n',
                    'a b</w>\n',
                ])
            cfg = OptimalBPEConfig(bpe_codes=path)
            encoder = OptimalBPE(cfg)
        finally:
            os.remove(path)

        sentence = "abccbaba bcaab\n"
        tokenized = encoder.encode(sentence)
        self.assertEqual("ab@@ c@@ cb@@ ab@@ a bc@@ a@@ ab", tokenized)


if __name__ == '__main__':
    unittest.main()
