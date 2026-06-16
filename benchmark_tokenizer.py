#!/usr/bin/env python3
"""Benchmark the tokenizer throughput."""

import time
import os
import argparse
from cs336_basics.tokenizer import Tokenizer


def benchmark_tokenizer(
    vocab_path: str,
    merges_path: str,
    input_file: str,
    sample_size_mb: int = 100,
):
    """Benchmark tokenizer throughput on a sample of data."""
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_path=vocab_path,
        merges_path=merges_path,
        special_tokens=["<|endoftext|>"],
    )
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    
    # Read sample from input file
    file_size = os.path.getsize(input_file)
    sample_bytes = sample_size_mb * 1024 * 1024
    
    print(f"\nReading {sample_size_mb} MB sample from {input_file}...")
    print(f"  Total file size: {file_size / 1e9:.2f} GB")
    
    with open(input_file, encoding="utf-8") as f:
        text = f.read(sample_bytes)
    
    # Trim to last complete line to avoid partial characters
    last_newline = text.rfind("\n")
    if last_newline > 0:
        text = text[:last_newline + 1]
    
    actual_bytes = len(text.encode("utf-8"))
    print(f"  Sample size (actual): {actual_bytes / 1e6:.2f} MB")
    
    # Benchmark encoding
    print(f"\nBenchmarking encoding...")
    start_time = time.time()
    tokens = tokenizer.encode(text)
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    bytes_per_second = actual_bytes / elapsed_time
    tokens_per_second = len(tokens) / elapsed_time
    bytes_per_token = actual_bytes / len(tokens) if tokens else float('inf')
    tokens_per_byte = len(tokens) / actual_bytes if actual_bytes else 0.0
    
    print(f"  Time: {elapsed_time:.2f} seconds")
    print(f"  Tokens: {len(tokens):,}")
    print(f"  Throughput: {bytes_per_second / 1e6:.2f} MB/s")
    print(f"  Throughput: {bytes_per_second / 1e9:.3f} GB/s")
    print(f"  Tokens/sec: {tokens_per_second:,.0f}")
    print(f"  Compression: {bytes_per_token:.3f} bytes/token ({tokens_per_byte:.6f} tokens/byte)")
    
    # Estimate time for full datasets
    print(f"\n{'='*60}")
    print("Estimated tokenization times:")
    print(f"{'='*60}")
    
    datasets = [
        ("TinyStories (2.1 GB)", 2.1),
        ("OpenWebText (11 GB)", 11.0),
        ("The Pile (825 GB)", 825.0),
    ]
    
    for name, size_gb in datasets:
        total_bytes = size_gb * 1e9
        total_seconds = total_bytes / bytes_per_second
        
        hours = total_seconds / 3600
        minutes = (total_seconds % 3600) / 60
        
        if hours >= 1:
            time_str = f"{hours:.1f} hours ({minutes:.0f} min)"
        else:
            time_str = f"{minutes:.1f} minutes"
        
        print(f"  {name:30s}: {time_str}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark tokenizer throughput")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab .json file")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges .txt file")
    parser.add_argument("--input", type=str, required=True, help="Input text file")
    parser.add_argument("--sample-size-mb", type=int, default=200, help="Sample size in MB")
    
    args = parser.parse_args()
    
    benchmark_tokenizer(
        vocab_path=args.vocab,
        merges_path=args.merges,
        input_file=args.input,
        sample_size_mb=args.sample_size_mb,
    )
