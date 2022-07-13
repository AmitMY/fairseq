# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Dict, List
from dataclasses import dataclass, field

from fairseq.data.encoders import register_bpe
from .subword_nmt_bpe import SubwordNMTBPE, SubwordNMTBPEConfig

import numpy as np
import itertools as it


class OptimalBPETokenizer:
    def __init__(self, vocabulary: Dict[str, int], separator: str, dropout: int = 0):
        self.vocabulary = vocabulary
        self.separator = separator

        self.cache = {}
        self.dropout = dropout

    def semi_markov_segment(self, word: str, dropout=0):
        """
        semi-Markov dynamic program for shortest segmentation with max frequency
        TODO: apply dropout, by removing some "codes" from the vocabulary, and unk-segmentation
        """

        N = len(word)

        # dynamic-programming array
        𝜷 = np.ones((N + 1, 2), dtype=int) * N
        𝜷[0] = 0
        𝜷[:, 1] = 0

        # backpointers
        bp = {0: -1}

        for m in range(N + 1):
            for n in range(m):
                segment = word[n:m]
                if segment in self.vocabulary:
                    if 𝜷[n, 0] + 1 < 𝜷[m, 0]:
                        𝜷[m, 0] = 𝜷[n, 0] + 1
                        𝜷[m, 1] = 𝜷[n, 1] + self.vocabulary[segment]
                        bp[m] = n
                    elif 𝜷[m, 0] == 𝜷[n, 0] + 1:
                        𝜷[m, 1] = max(𝜷[m, 1], 𝜷[n, 1] + self.vocabulary[segment])
                        bp[m] = n

        # extract backpointers
        ptr = N
        segmentation = []
        while ptr != 0:
            segmentation = [word[bp[ptr]:ptr]] + segmentation
            ptr = bp[ptr]

        # make sure we didn't fuck it
        # assert 𝜷[N, 1] == sum(map(lambda x: self.vocabulary[x], segmentation))
        assert "".join(segmentation) == word

        return segmentation, 𝜷[N, 0], 𝜷[N, 1]

    def segment_word(self, word: str):
        """
        segment a single word, aware of cache
        """
        if not self.dropout and word in self.cache:
            return self.cache[word]

        self.cache[word], _, _ = self.semi_markov_segment(word, dropout=self.dropout)
        return self.cache[word]

    def segment_words(self, words: List[str], dropout=0) -> List[str]:
        """segment a sequence of words with BPE encoding"""
        for word in words:
            segments = self.segment_word(word)
            for segment in segments[:-1]:
                yield segment + self.separator

            yield segments[-1]

    def segment(self, sentence: str):
        """segmenta  single sentence (whitespace-tokenized string) with BPE encoding"""
        segments = self.segment_words(sentence.strip('\r\n ').split())
        return ' '.join(segments)

    def brute_segment(self, sentence):
        """
        Brute-force method to compute the shortest segmentation
        such that every segment is in the vocabulary
        """
        N = len(sentence)
        min_len, max_freq = N, 0

        for mask in it.product([0, 1], repeat=N):
            segmentation = []
            cur = ""
            for i, c in zip(mask, sentence):
                if i == 1:
                    segmentation.append(cur)
                    cur = c
                else:
                    cur += c
            if cur != "":
                segmentation.append(cur)

            # check whether each segment is in the vocabulary
            bad = False
            for segment in segmentation:
                if segment not in self.vocabulary:
                    bad = True
            if not bad:
                freq = sum(map(lambda x: self.vocabulary[x], segmentation))
                if len(segmentation) < min_len:
                    min_len = len(segmentation)
                    max_freq = freq
                elif len(segmentation) == min_len:
                    max_freq = max(max_freq, freq)

            assert "".join(segmentation) == sentence, print(segmentation, mask)

        return min_len, max_freq


@dataclass
class OptimalBPEConfig(SubwordNMTBPEConfig):
    bpe_dropout: int = field(default=0, metadata={"help": "BPE dropout (0 to 1)"})


@register_bpe("optimal", dataclass=OptimalBPEConfig)
class OptimalBPE(SubwordNMTBPE):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.tokenizer = OptimalBPETokenizer(self.get_vocabulary(), self.bpe.separator)

    def get_vocabulary(self, whitespace_token='✡'):
        vocabulary = {b + p: 1 / (f + 1) for (b, p), f in self.bpe.bpe_codes.items()}
        vocabulary = {k.replace('</w>', whitespace_token): v for k, v in vocabulary.items()}
        # Vocabulary does not include single characters, or single characters with whitespace token!
        for w in list(vocabulary.keys()):
            for c in w:
                vocabulary[c] = vocabulary[c + whitespace_token] = 0

        return vocabulary

    def encode(self, x: str) -> str:
        return self.tokenizer.segment(x)
