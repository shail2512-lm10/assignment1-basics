from collections.abc import Iterable, Iterator
import json
import pathlib
from cs336_basics.utils import get_mappings
from cs336_basics.pretokenization import PreTokenizer


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        

        if self.special_tokens is not None:
            for token in self.special_tokens:
                if token.encode("utf-8") not in self.vocab.values():
                    self.vocab[len(self.vocab)] = token.encode("utf-8")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.merges_dict = {tup: idx for idx, tup in enumerate(self.merges)}
        self._word_cache: dict[bytes, tuple[int, ...]] = {}


    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        _, u2b = get_mappings()
        # creates vocab from vocab_path json file
        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)
            
        vocab = {}
        for pretty_string, token_id in data.items():
            # Convert each character back to its original byte value
            byte_list = [u2b[char] for char in pretty_string]
            vocab[token_id] = bytes(byte_list)
        
        # creates merges from merges_path text file
        merges = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                
                # Split the two printable tokens
                parts = line.strip().split(' ')
                if len(parts) != 2:
                    continue
                
                # Convert each printable token back to raw bytes
                pair_bytes = []
                for part in parts:
                    byte_seq = bytes([u2b[char] for char in part])
                    pair_bytes.append(byte_seq)
                    
                merges.append(tuple(pair_bytes))

        return cls(vocab, merges, special_tokens)

    
    def _encode_word(self, word: bytes) -> tuple[int, ...]:
        if word in self._word_cache:
            return self._word_cache[word]

        if word in self.inverse_vocab:
            token_ids = (self.inverse_vocab[word],)
            self._word_cache[word] = token_ids
            return token_ids

        word_parts = [bytes([b]) for b in word]

        while len(word_parts) > 1:
            best_pair_idx = None
            best_rank = float("inf")

            for idx, pair in enumerate(zip(word_parts, word_parts[1:])):
                rank = self.merges_dict.get(pair, float("inf"))
                if rank < best_rank:
                    best_pair_idx = idx
                    best_rank = rank

            if best_pair_idx is None or best_rank == float("inf"):
                break

            word_parts[best_pair_idx] = word_parts[best_pair_idx] + word_parts[best_pair_idx + 1]
            del word_parts[best_pair_idx + 1]

        token_ids = tuple(self.inverse_vocab[w] for w in word_parts)
        self._word_cache[word] = token_ids
        return token_ids

    
    def encode(self, text: str) -> list[int]:
        pre_tokenized_word_bytes = PreTokenizer.pretokenize_encode(text, self.special_tokens, desired_num_chunks=1, num_workers=1)
        # print("Pre-tokenized byte sequences:", pre_tokenized_word_bytes)  # Debugging statement
        # Convert pre-tokenized byte sequences to token IDs
        encoded_text = []
        for word in pre_tokenized_word_bytes:
            encoded_text.extend(self._encode_word(word))


        return encoded_text

    def decode(self, ids: list[int]) -> str:
        """Convert a sequence of token IDs back to a string.

        The previous implementation decoded each token separately which could
        split multi-byte UTF-8 sequences (e.g. emoji) across multiple calls to
        ``bytes.decode``.  That behavior resulted in replacement characters (�)
        when the token boundaries did not align with Unicode code point
        boundaries.  Instead we concatenate **all** of the raw bytes first and
        then perform a single UTF‑8 decode.  This guarantees that multi‑byte
        characters are handled correctly, and mirrors the behavior of the
        reference tokenizer.
        """

        # build the full byte sequence before decoding so that sequences which
        # span token boundaries remain intact.  We still perform a vocab check
        # early so invalid IDs are caught quickly.
        byte_chunks: list[bytes] = []
        for num in ids:
            if num not in self.vocab:
                raise ValueError(f"Token ID {num} not found in vocab.")
            byte_chunks.append(self.vocab[num])

        full_bytes = b"".join(byte_chunks)
        # ``errors="replace"`` is only used as a last-resort fallback; under
        # normal operation the bytes should form valid UTF-8 because the
        # vocabulary is derived from unicode text.
        return full_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Yield token ids from a sequence of strings supplied by *iterable*.

        This mirrors :meth:`encode` but allows callers to process input
        lazily.  We simply iterate over ``iterable``, convert non-strings to
        ``str`` (some iterables like file objects produce bytes) and then
        yield each token id returned by :meth:`encode` one at a time.
        """

        for piece in iterable:
            if not isinstance(piece, str):
                piece = str(piece)

            yield from self.encode(piece)


if __name__ == "__main__":
    parent_path = pathlib.Path(__file__).parent.parent
    tokenizer = Tokenizer.from_files(
        vocab_path=parent_path / "data/vocab.json",
        merges_path=parent_path / "data/merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    with open(parent_path / "tests/fixtures/tinystories_sample.txt") as f:
        content = f.read()

    encoded = tokenizer.encode(content)
    decoded = tokenizer.decode(encoded)
    assert decoded == content
