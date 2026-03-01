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

    
    def encode(self, text: str) -> list[int]:
        pre_tokenized_word_bytes = PreTokenizer.pretokenize_encode(text, self.special_tokens, desired_num_chunks=64)
        # print("Pre-tokenized byte sequences:", pre_tokenized_word_bytes)  # Debugging statement
        # Convert pre-tokenized byte sequences to token IDs
        encoded_text = []
        for word in pre_tokenized_word_bytes:
            if word in self.inverse_vocab:
                token_id = self.inverse_vocab[word]
                encoded_text.append(token_id)
            else:
                word_byte = [bytes([b]) for b in word]
                pairs = list(zip(word_byte[:], word_byte[1:]))
                for pair in pairs:
                    if pair in self.merges:
                        is_in_merges = True
                        break
                    else:
                        is_in_merges = False

                while len(word_byte) > 1 and is_in_merges:
                
                    pairs_stats = {p: (id, self.merges_dict[p] if p in self.merges else float('inf')) for id, p in enumerate(pairs)}
                    merge_pair = min(pairs_stats.keys(), key=lambda k: pairs_stats[k][1])
                    merge_token, merge_id = merge_pair[0] + merge_pair[1], pairs_stats[merge_pair][0]
                    word_byte[merge_id] = merge_token
                    del word_byte[merge_id + 1]

                    pairs = list(zip(word_byte[:], word_byte[1:]))
                    for pair in pairs:
                        if pair in self.merges:
                            is_in_merges = True
                            break
                        else:
                            is_in_merges = False

                # Now convert the final remaining each byte sequence to token ID
                for w in word_byte:
                    if w in self.inverse_vocab:
                        encoded_text.append(self.inverse_vocab[w])
                    else:
                        raise ValueError(f"Unknown token: {w}")


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