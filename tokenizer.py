# Taken from llama code and lightly modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according
# to the terms of the Llama 2 Community License Agreement.

"""
Tokenizer implementation for Llama 2 models using SentencePiece.
Provides encoding/decoding functionality and binary export for C inference.
"""

import os
import struct
import argparse
from typing import List

from sentencepiece import SentencePieceProcessor

TOKENIZER_MODEL = "tokenizer.model"  # the llama sentencepiece tokenizer model


class Tokenizer:
    """
    SentencePiece tokenizer wrapper for Llama 2 models.
    Handles encoding text to token IDs and decoding token IDs back to text.
    Also supports exporting the tokenizer to a binary format for C inference.
    """

    def __init__(self, tokenizer_model=None):
        model_path = tokenizer_model if tokenizer_model else TOKENIZER_MODEL
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor()
        self.sp_model.Load(model_path)
        self.model_path = model_path

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        # print(f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}")
        assert self.sp_model.vocab_size() == self.sp_model.GetPieceSize()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encode a string into a list of token IDs.

        Args:
            s: The input string to encode
            bos: Whether to prepend the beginning-of-sequence token
            eos: Whether to append the end-of-sequence token

        Returns:
            List of token IDs
        """
        assert isinstance(s, str)
        tokens = self.sp_model.Encode(s, out_type=int)
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            tokesn: List of token IDs to decode

        Returns:
            Decoded string
        """
        return self.sp_model.Decode(tokens)

    def export(self):
        """
        Export the tokenizer to a binary file format for use in C inference code.
        Converts the SentencePiece model to a simple binary format with token scores.
        """
        # get all the tokens (postprocessed) and their scores as floats
        tokens, scores = [], []
        for i in range(self.n_words):
            # decode the token and light postprocessing
            token = self.sp_model.IdToPiece(i)
            score = self.sp_model.GetScore(i)
            if i == self.bos_id:
                token = "\n<s>\n"
            elif i == self.eos_id:
                token = "\n</s>\n"
            token = token.replace(
                "▁", " "
            )  # sentencepiece uses this character as whitespace
            token_bytes = token.encode("utf-8")  # bytes of this token, utf-8 encoded

            tokens.append(token_bytes)
            scores.append(score)

        # record the max token length
        max_token_length = max(len(t) for t in tokens)

        # write to a binary file
        # the tokenizer.bin file is the same as .model file, but .bin
        tokenizer_bin = self.model_path.replace(".model", ".bin")
        with open(tokenizer_bin, "wb") as f:
            f.write(struct.pack("I", max_token_length))
            for b, score in zip(tokens, scores):
                f.write(struct.pack("fI", score, len(b)))
                f.write(b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--tokenizer-model", type=str, help="optional path to custom tokenizer "
    )
    args = parser.parse_args()

    t = Tokenizer(args.tokenizer_model)
    t.export()
