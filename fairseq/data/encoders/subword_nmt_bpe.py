# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from fairseq.dataclass import FairseqDataclass


@dataclass
class SubwordNMTBPEConfig(FairseqDataclass):
    bpe_codes: str = field(default="???", metadata={"help": "path to subword NMT BPE codes"})
    bpe_separator: str = field(default="@@", metadata={"help": "BPE separator"})
    bpe_vocab: Optional[str] = field(default=None, metadata={"help": "path to subword NMT BPE vocab"})


@register_bpe("subword_nmt", dataclass=SubwordNMTBPEConfig)
class SubwordNMTBPE(object):
    def __init__(self, cfg):
        if cfg.bpe_codes is None:
            raise ValueError("--bpe-codes is required for --bpe=subword_nmt")
        codes = file_utils.cached_path(cfg.bpe_codes)
        try:
            from subword_nmt import apply_bpe

            bpe_parser = apply_bpe.create_parser()
            bpe_raw_args = [
                "--codes",
                codes,
                "--separator",
                cfg.bpe_separator,
            ]
            if cfg.bpe_vocab is not None:
                bpe_raw_args.append("--vocabulary")
                bpe_raw_args.append(cfg.bpe_vocab)
            bpe_args = bpe_parser.parse_args(bpe_raw_args)
            self.bpe = apply_bpe.BPE(
                bpe_args.codes,
                bpe_args.merges,
                bpe_args.separator,
                bpe_args.vocabulary,
                bpe_args.glossaries,
            )
            self.bpe_symbol = bpe_args.separator + " "
        except ImportError:
            raise ImportError(
                "Please install subword_nmt with: pip install subword-nmt"
            )

    def encode(self, x: str) -> str:
        return self.bpe.process_line(x)

    def decode(self, x: str) -> str:
        return (x + " ").replace(self.bpe_symbol, "").rstrip()
