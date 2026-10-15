import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class CustomCollate:
    def __init__(self, device=None):
        self.device = device

    def __call__(self, batch):
        inputs, targets = zip(*batch)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)
        
        if self.device is not None:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        
        return inputs, targets

class TokenDataset(Dataset):
    """
    PyTorch Dataset that loads tokenized sequences from a file or numpy array.
    Creates (input, target) pairs for language modeling.
    """
    
    def __init__(
        self,
        data: str | np.ndarray,
        context_length: int,
        dtype=np.uint32,
        vocab_size: int | None = None,
    ):
        """
        Args:
            data: Either a filepath (str) to load with memmap, or a numpy array of token IDs
            context_length: Length of input sequences
            dtype: Data type for memmap (only used if data is a filepath)
            vocab_size: If set, validate that all token IDs fit the model vocabulary
        """
        self.context_length = context_length
        
        # Handle both filepath and numpy array inputs.
        # .npy files include a header, so they must be opened with np.load rather
        # than raw np.memmap; otherwise header bytes become bogus token IDs.
        if isinstance(data, str):
            data_path = Path(data)
            if data_path.suffix == ".npy":
                self.data = np.load(data_path, mmap_mode="r")
            else:
                self.data = np.memmap(data_path, dtype=dtype, mode='r')
        else:
            self.data = data
            
        # Number of valid starting positions for sequences
        # We need at least context_length + 1 tokens
        self.num_samples = len(self.data) - context_length
        
        if self.num_samples <= 0:
            raise ValueError(
                f"Data length ({len(self.data)}) must be greater than "
                f"context_length ({context_length})"
            )

        if vocab_size is not None:
            min_token = int(np.min(self.data))
            max_token = int(np.max(self.data))
            if min_token < 0 or max_token >= vocab_size:
                raise ValueError(
                    "Tokenized data contains token IDs outside the configured "
                    f"vocabulary: min={min_token}, max={max_token}, "
                    f"vocab_size={vocab_size}. Use the vocab size from the "
                    "tokenizer that produced this data, or retokenize with the "
                    "configured tokenizer."
                )
    
    def __len__(self) -> int:
        """Returns the number of available samples"""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Returns a sequential (input, target) pair at the given index.
        
        Args:
            idx: Index of the sample (0 to num_samples - 1)
        
        Returns:
            Tuple of (input_ids, target_ids) as torch tensors
        """
        # Get the input sequence starting at index idx
        input_ids = self.data[idx : idx + self.context_length]
        # Target is shifted by 1 (next token prediction)
        target_ids = self.data[idx + 1 : idx + self.context_length + 1]
        
        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long)
        target_tensor = torch.tensor(target_ids, dtype=torch.long)
        
        return input_tensor, target_tensor


def get_batch_dataloader(
    data: str | np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle: bool = True,
    vocab_size: int | None = None,
) -> DataLoader:
    """
    Create a PyTorch DataLoader for tokenized sequences.
    
    Args:
        data: Either a filepath (str) to load with memmap, or a numpy array of token IDs
        batch_size: Number of sequences per batch
        context_length: Length of each sequence
        device: PyTorch device to place tensors on when loading without worker processes.
            With num_workers > 0, batches are returned on CPU and should be moved to
            the target device in the main process.
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        shuffle: Whether to shuffle the batches (default: True for random sampling)
        vocab_size: If set, validate token IDs against this vocabulary size
    
    Returns:
        DataLoader that yields (batch_inputs, batch_targets) batches
    """
    dataset = TokenDataset(data, context_length, vocab_size=vocab_size)
    
    collate_device = None if num_workers > 0 else device
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Randomly shuffle batches by default
        collate_fn=CustomCollate(collate_device),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return dataloader
