"""BPE implementation with inverted index for faster merges."""

from __future__ import annotations
import os
from typing import BinaryIO
from multiprocessing import Pool
import pathlib
import json
from functools import lru_cache
from cs336_basics.utils import get_mappings
from cs336_basics.pretokenization import PreTokenizer

import regex

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]
PairCounts = dict[tuple[bytes, bytes], int]
PairToWords = dict[tuple[bytes, bytes], set[int]]  # NEW: inverted index

Words = list[list[bytes]]
WordFreq = list[int]

WordCounts = dict[tuple[bytes, ...], int]

GPT2_REGEX = regex.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)



class TrainTokenizer:
    """BPE Tokenizer trainer wrapping all training functions."""
    def __init__(self, pre_tokenizer: PreTokenizer) -> None:
        self.pre_tokenizer = pre_tokenizer
        

    @staticmethod
    def save_vocab_and_merges(output_path, vocab: Vocab, merges: Merges) -> None:
        """
        vocab_dict: dict(int, bytes) -> {0: b'\x00', 256: b'th', ...}
        Saves as JSON: { "0": "Ā", "256": "th" }

        merges: list of tuples -> [(b't', b'h'), (b'th', b'e'), ...]
        Saves as:
        t h
        th e
        (using printable mapping)
        """
        b2u, _ = get_mappings()
        serialized = {}
        
        for token_id, byte_seq in vocab.items():
            # Convert each byte in the sequence to its mapped Unicode character
            pretty_string = "".join(b2u[b] for b in byte_seq)
            serialized[pretty_string] = token_id
            
        with open(output_path / "vocab.json", 'w', encoding='utf-8') as f:
            json.dump(serialized, f, ensure_ascii=False, indent=4)

        with open(output_path / "merges.txt", 'w', encoding='utf-8') as f:
            # Optional: OpenAI files often start with a version/comment line
            # f.write("#version: 0.2\n")
            for pair in merges:
                # Map each 'side' of the merge to its printable representation
                parent_1 = "".join(b2u[b] for b in pair[0])
                parent_2 = "".join(b2u[b] for b in pair[1])
                f.write(f"{parent_1} {parent_2}\n")

    

    @staticmethod
    def find_most_frequent_pair(
        pair_counts: PairCounts,
    ) -> tuple[bytes, bytes] | None:
        """Find the most frequent pair - now O(n) over pairs, not over all tokens."""
        if not pair_counts:
            return None
        return max(pair_counts, key=lambda pair: (pair_counts[pair], pair[0], pair[1]))

    @staticmethod
    def get_word_pairs(word: list[bytes]) -> list[tuple[bytes, bytes]]:
        """Get all adjacent pairs in a word."""
        return [(word[i], word[i + 1]) for i in range(len(word) - 1)]

    @staticmethod
    def apply_merge(
        words: Words,
        word_freq: WordFreq,
        pair: tuple[bytes, bytes],
        pair_counts: PairCounts,
        pair_to_words: PairToWords,
    ) -> None:
        """Apply merge and update pair_counts and pair_to_words by comparing old/new pairs."""
        merged = pair[0] + pair[1]
        affected_word_ids = pair_to_words.pop(pair, set())
        del pair_counts[pair]

        for word_id in affected_word_ids:
            word = words[word_id]
            freq = word_freq[word_id]

            old_pairs = TrainTokenizer.get_word_pairs(word)

            i = 0
            while i < len(word) - 1:
                if word[i] == pair[0] and word[i + 1] == pair[1]:
                    word[i] = merged
                    del word[i + 1]
                else:
                    i += 1

            new_pairs = TrainTokenizer.get_word_pairs(word)

            for p in old_pairs:
                if p == pair:
                    continue
                pair_counts[p] -= freq
                if pair_counts[p] == 0:
                    del pair_counts[p]
                    pair_to_words.pop(p, None)
                else:
                    pair_to_words[p].discard(word_id)

            for p in new_pairs:
                pair_counts[p] = pair_counts.get(p, 0) + freq
                if p not in pair_to_words:
                    pair_to_words[p] = set()
                pair_to_words[p].add(word_id)


    def train_bpe(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str],
    ) -> tuple[Vocab, Merges]:
        """Train a BPE tokenizer and return vocabulary and merges."""
        words, word_freq, pair_counts, pair_to_words = self.pre_tokenizer.pretokenize(input_path, special_tokens, 4)
        vocab = TrainTokenizer.build_initial_vocab(special_tokens)
        merges: Merges = []

        num_merges = vocab_size - len(vocab)

        for _ in range(num_merges):
            best_pair = TrainTokenizer.find_most_frequent_pair(pair_counts)
            if best_pair is None:
                break
            merges.append(best_pair)
            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            TrainTokenizer.apply_merge(words, word_freq, best_pair, pair_counts, pair_to_words)

        return vocab, merges

    @staticmethod
    def build_initial_vocab(special_tokens: list[str]) -> Vocab:
        """Build initial vocabulary with 256 single bytes + special tokens."""
        vocab = {}
        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")

        n = len(vocab)
        for i in range(256):
            vocab[i + n] = bytes([i])

        return vocab

if __name__ == "__main__":
    parent_path = pathlib.Path(__file__).parent.parent
    train_tokenizer = TrainTokenizer(PreTokenizer)
    vocab, merges = train_tokenizer.train_bpe(
        input_path=parent_path / "tests/fixtures/corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"]
    )
    TrainTokenizer.save_vocab_and_merges(parent_path / "data", vocab, merges)


