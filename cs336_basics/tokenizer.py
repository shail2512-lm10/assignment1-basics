from collections.abc import Iterable, Iterator
import json
import pathlib
from cs336_basics.utils import get_mappings
from cs336_basics.pretokenization import PreTokenizer
from multiprocessing import Pool


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


    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: list[str] | None = None):
        _, u2b = get_mappings()
        # creates vocab from vocab_path json file
        with open(vocab_path, "r", encoding="utf-8") as f:
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
                if len(parts) != 2: continue
                
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
        for idx, word in enumerate(pre_tokenized_word_bytes):
            # print(f"Processing word {idx}: {word}")  # Debugging statement
            if word in self.inverse_vocab:
                token_id = self.inverse_vocab[word]
                encoded_text.append(token_id)
            else:
                word_byte = [bytes([b]) for b in word]
                # print("word_byte:", word_byte)
                i = 0
                while i < len(word_byte) - 1:
                    if word[i:] in self.inverse_vocab:
                        encoded_text.append(self.inverse_vocab[word[i:]])
                        i += 1
                        break
                    pair = (word_byte[i], word_byte[i + 1])
                    if pair in self.merges:
                        # Merge the pair
                        word_byte[i] = word_byte[i] + word_byte[i + 1]
                        del word_byte[i + 1]
                    else:
                        i += 1
                # Now convert each remaining byte sequence to token ID
                for w in word_byte:
                    if w in self.inverse_vocab:
                        encoded_text.append(self.inverse_vocab[w])
                    else:
                        raise ValueError(f"Unknown token: {w}")


        return encoded_text

    def decode(self, ids: list[int]) -> str:
        text = ""
        for num in ids:
            text += self.vocab[num].decode("utf-8", errors="replace")
        return text

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            chunk = (text, self.special_tokens)
            with Pool() as pool:
                for result in pool.map(PreTokenizer._pretokenize_chunk_encode, chunk):
                    yield from result


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