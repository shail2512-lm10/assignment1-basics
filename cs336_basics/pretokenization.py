import pathlib
import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from collections import Counter

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
    def _find_chunk_boundaries(
        file: BinaryIO | None = None,
        desired_num_chunks: int = 4,
        split_special_token: bytes | str = b"",
        text: str | None = None
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        # assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        if file is not None:
            file.seek(0, os.SEEK_END)
            total_size = file.tell()
            file.seek(0)

            chunk_size = total_size // desired_num_chunks

        if text is not None:
            total_size = len(text)
            chunk_size = total_size // desired_num_chunks
        

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = total_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            if file is not None:
                file.seek(initial_position) # Start at boundary guess
            while True:
                if file is not None:
                    mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                if text is not None:
                    mini_chunk = text[initial_position:initial_position + mini_chunk_size]

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"" or mini_chunk == "":
                    chunk_boundaries[bi] = total_size
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
    def _pretokenize_chunk_train(args: tuple[str, int, int, list[str]]) -> WordCounts:
        """Process a single chunk of the file."""
        input_path, start, end, special_tokens = args

        with open(input_path, "rb") as f:
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8")

        corpus_split = [chunk]

        if special_tokens:
            # sort tokens by length to prefer longer (overlapping) matches first
            tokens_sorted = sorted(special_tokens, key=len, reverse=True)
            pattern = (
                r"(" + "|".join(re.escape(token) for token in tokens_sorted) + r")"
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
    def pretokenize_train(
        input_path: str, special_tokens: list[str], desired_num_chunks: int, num_workers: int | None = None
    ) -> tuple[Words, WordFreq, PairCounts, PairToWords]:
        """Pre-tokenize input, returning words, frequencies, pair counts, and inverted index."""
        split_token = special_tokens[0].encode("utf-8") if special_tokens else b"\n"
        with open(input_path, "rb") as f:
            boundaries = PreTokenizer._find_chunk_boundaries(file=f, desired_num_chunks=desired_num_chunks, split_special_token=split_token)

        chunks = [
            (input_path, boundaries[i], boundaries[i + 1], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        with Pool(num_workers) as pool:
            results = pool.map(PreTokenizer._pretokenize_chunk_train, chunks)

        return PreTokenizer._merge_word_counts(results)
    
    @staticmethod
    def _merge_word_counts(
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
    

    @staticmethod
    def pretokenize_encode(text: str, special_tokens: list[str], desired_num_chunks: int, num_workers: int | None = None) -> list[bytes]:
        """Pre-tokenize a single string into a list of byte tokens."""
        split_token = "<|endoftext|>"
        boundaries = PreTokenizer._find_chunk_boundaries(text=text, desired_num_chunks=desired_num_chunks, split_special_token=split_token)

        chunks = [
            (text[boundaries[i]:boundaries[i + 1]], special_tokens)
            for i in range(len(boundaries) - 1)
        ]

        with Pool(processes=num_workers) as pool:
            results = pool.map(PreTokenizer._pretokenize_chunk_encode, chunks)

        pre_tokenized_word_bytes = []
        for result in results:
            pre_tokenized_word_bytes.extend(result)
        return pre_tokenized_word_bytes
    

    @staticmethod
    def _pretokenize_chunk_encode(args: tuple[str, list[str]]) -> list[bytes]:
        """Pre-tokenize a single chunk into a list of byte tokens."""
        chunk, special_tokens = args

        chunk_pre_tokenized_word_bytes = []

        if special_tokens:
            # prefer longer tokens first to handle overlapping cases
            tokens_sorted = sorted(special_tokens, key=len, reverse=True)
            pattern = (
                r"(" + "|".join(re.escape(token) for token in tokens_sorted) + r")"
            )
            corpus_split = re.split(pattern, chunk)
            corpus_split = [x for x in corpus_split if x]

            for corpus in corpus_split:
                if corpus in special_tokens:
                    chunk_pre_tokenized_word_bytes.append(corpus.encode("utf-8"))
                    continue
                matches = re.finditer(GPT2_REGEX, corpus)
                for m in matches:
                    chunk_pre_tokenized_word_bytes.append(m.group().encode("utf-8"))
        else:
            corpus_split = chunk
            matches = re.finditer(GPT2_REGEX, corpus_split)
            for m in matches:
                chunk_pre_tokenized_word_bytes.append(m.group().encode("utf-8"))

        # return the list so multiprocessing can collect results
        return chunk_pre_tokenized_word_bytes


if __name__ == "__main__":
    input_path = pathlib.Path(__file__).parent.parent / "tests/fixtures/tinystories_sample.txt"
    special_tokens = ["<|endoftext|>"]
    text = input_path.read_text(encoding="utf-8")
    # words, word_freq, pair_counts, pair_to_words = PreTokenizer.pretokenize_train(input_path, special_tokens, 4)
    word_bytes = PreTokenizer.pretokenize_encode(text, special_tokens, 4)
    print(word_bytes)
    # print(words)
    # print("--------------------")
    # print(word_freq)
    # print("---------------")
    # print(pair_counts)
    # print("---------------")
    # print(pair_to_words)
    