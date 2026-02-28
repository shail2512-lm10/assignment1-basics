import pathlib
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from functools import partial
from collections import Counter
import pathlib

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]
PairCounts = dict[tuple[bytes, bytes], int]
PairToWords = dict[tuple[bytes, bytes], set[int]]  # NEW: inverted index

Words = list[list[bytes]]
WordFreq = list[int]

WordCounts = dict[tuple[bytes, ...], int]

GPT2_REGEX = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class PreTokenizer:
    @staticmethod
    def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> WordCounts:
        """Process a single chunk of the file."""
        input_path, start, end, special_tokens = args

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")

        corpus_split = [chunk]

        if special_tokens:
            pattern = (
                r"(" + "|".join(re.escape(token) for token in special_tokens) + r")"
            )
            corpus_split = re.split(pattern, chunk)
            corpus_split = [x for x in corpus_split if x]

        chunk_pre_tokenized_texts = []
        for corpus in corpus_split:
            if corpus in special_tokens:
                continue
            matches = re.finditer(GPT2_REGEX, corpus)
            for m in matches:
                chunk_pre_tokenized_texts.extend([m.group().encode("utf-8")])
        word_count = Counter(chunk_pre_tokenized_texts)

        return word_count

    @staticmethod
    def pretokenize(
        input_path: str, special_tokens: list[str], desired_num_chunks: int, num_workers: int | None = None
    ) -> tuple[Words, WordFreq, PairCounts, PairToWords]:
        """Pre-tokenize input, returning words, frequencies, pair counts, and inverted index."""
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        with open(input_path, "rb") as f:
            boundaries = PreTokenizer.find_chunk_boundaries(f, desired_num_chunks, split_token)

        chunks = [
            (input_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        with Pool(num_workers) as pool:
            results = pool.map(PreTokenizer.pretokenize_chunk, chunks)

        return PreTokenizer.merge_word_counts(results)
    
    @staticmethod
    def merge_word_counts(
        counts_list: list[WordCounts],
    ) -> tuple[Words, WordFreq, PairCounts, PairToWords]:
        """Merge multiple WordCounts dicts into one, building pair counts and inverted index."""

        word_to_id: dict[tuple[bytes, ...], int] = {}
        words: Words = []
        word_freq: WordFreq = []

        # First pass: merge word counts (same as before)
        for counts in counts_list:
            for word_bytes, count in counts.items():
                if word_bytes in word_to_id:
                    word_id = word_to_id[word_bytes]
                    word_freq[word_id] += count
                else:
                    word_id = len(words)
                    word_to_id[word_bytes] = word_id
                    words.append([bytes([b]) for b in word_bytes])
                    word_freq.append(count)

        # NEW: Build initial pair_counts and pair_to_words
        pair_counts: PairCounts = {}
        pair_to_words: PairToWords = {}

        for word_id, word in enumerate(words):
            freq = word_freq[word_id]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                # Update count
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
                # Update inverted index
                if pair not in pair_to_words:
                    pair_to_words[pair] = set()
                pair_to_words[pair].add(word_id)

        return words, word_freq, pair_counts, pair_to_words


if __name__ == "__main__":
    input_path = pathlib.Path(__file__).parent.parent / "tests/fixtures/tinystories_sample.txt"
    special_tokens = ["<|endoftext|>"]
    words, word_freq, pair_counts, pair_to_words = PreTokenizer.pretokenize(input_path, special_tokens, 4)
    print(words)
    print("--------------------")
    print(word_freq)
    print("---------------")
    print(pair_counts)
    print("---------------")
    print(pair_to_words)
    