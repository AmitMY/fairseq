# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field

from fairseq.data.encoders import register_bpe
from fairseq.data.encoders.subword_nmt_bpe import SubwordNMTBPE, SubwordNMTBPEConfig


@dataclass
class OptimalBPEConfig(SubwordNMTBPEConfig):
    bpe_dropout: int = field(default=0, metadata={"help": "BPE dropout (0 to 1)"})


@register_bpe("optimal", dataclass=OptimalBPEConfig)
class OptimalBPE(SubwordNMTBPE):
    def __init__(self, cfg):
        super().__init__(cfg)

        try:
            from better_bpe import OptimalBPETokenizer
        except ImportError:
            raise ImportError(
                "Please install better_bpe with: pip install better-bpe"
            )

        self.tokenizer = OptimalBPETokenizer(self.get_vocabulary(), self.bpe.separator)

    def get_vocabulary(self):
        vocabulary = {b + p: 1 / (f + 1) for (b, p), f in self.bpe.bpe_codes.items()}
        vocabulary = {k.replace('</w>', ' '): v for k, v in vocabulary.items()}
        return vocabulary

    def encode(self, x: str) -> str:
        return self.tokenizer.segment(x)
