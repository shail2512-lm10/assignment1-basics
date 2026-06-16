import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Direct wget-compatible URL to owt_train.txt on HuggingFace.
# Replace with the exact resolve/main URL for your dataset file.
DATA_URL = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz"

VOCAB_SIZE      = 32_000
SPECIAL_TOKENS  = ["<|endoftext|>"]

# Paths inside the container
CONTAINER_DATA_PATH  = "/data/owt_train.txt"
CONTAINER_DATA_GZ_PATH = "/data/owt_train.txt.gz"
CONTAINER_OUTPUT_DIR = "/outputs"

# 128 GB is comfortable for 32k vocab on ~11 GB of raw text.
# The pretokenizer holds word-freq dicts in RAM; BPE merge loop is O(vocab_size).
MEMORY_GB       = 128
CPU_COUNT       = 16      # pretokenize_train uses 4 workers per your call, extra headroom is free
TIMEOUT_SECONDS = 60 * 60 * 12   # 6 h ceiling; typical run is 1-3 h

# ---------------------------------------------------------------------------
# Modal app
# ---------------------------------------------------------------------------

app = modal.App("bpe-trainer")

# Persistent volume — survives after the container exits so you can re-download
# outputs without re-running training.
output_volume = modal.Volume.from_name("bpe-outputs", create_if_missing=True)
data_volume = modal.Volume.from_name("bpe-data", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("wget", "gzip")
    .pip_install("regex", "tqdm")   # deps used by cs336_basics
    # Copies your entire local project (including cs336_basics/) into /app.
    # Make sure you run `modal run` from the root of your cs336 repo.
    .add_local_dir("cs336_basics", remote_path="/app/cs336_basics")
)

# ---------------------------------------------------------------------------
# Remote training function
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    cpu=CPU_COUNT,
    memory=MEMORY_GB * 1024,          # Modal takes MB
    timeout=TIMEOUT_SECONDS,
    nonpreemptible=True,
    volumes={CONTAINER_OUTPUT_DIR: output_volume, "/data": data_volume},
)
def train_tokenizer():
    import os
    import sys
    import subprocess
    import time
    import pathlib

    # Make sure the local repo is importable
    sys.path.insert(0, "/app")

    from cs336_basics.pretokenization import PreTokenizer
    from cs336_basics.train_bpe import TrainTokenizer

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------

    if os.path.exists(CONTAINER_DATA_PATH):
        size_gb = os.path.getsize(CONTAINER_DATA_PATH) / 1e9
        print(f"[1/3] Data already exists ({size_gb:.2f} GB), skipping download.")
    else:
        print("[1/3] Downloading data ...")
        t0 = time.time()
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", CONTAINER_DATA_GZ_PATH, DATA_URL],
            check=True,
        )
        print(f"      Downloaded in {time.time() - t0:.1f}s, decompressing ...")
        subprocess.run(
            ["gunzip", "-f", CONTAINER_DATA_GZ_PATH],
            check=True,
        )
        size_gb = os.path.getsize(CONTAINER_DATA_PATH) / 1e9
        print(f"      Decompressed to {size_gb:.2f} GB")
    

    # ------------------------------------------------------------------
    # 2. Train BPE
    # ------------------------------------------------------------------
    print(f"[2/3] Training BPE (vocab_size={VOCAB_SIZE}) ...")
    t1 = time.time()

    trainer = TrainTokenizer(PreTokenizer)
    vocab, merges = trainer.train_bpe(
        input_path=pathlib.Path(CONTAINER_DATA_PATH),
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        num_workers=14,
        num_chunks=56,
    )

    elapsed = (time.time() - t1) / 60
    print(f"      Done in {elapsed:.1f} min — {len(vocab)} tokens, {len(merges)} merges")

    # ------------------------------------------------------------------
    # 3. Save to Modal Volume using your existing save method
    # ------------------------------------------------------------------
    print("[3/3] Saving vocab.json + merges.txt ...")

    output_path = pathlib.Path(CONTAINER_OUTPUT_DIR)
    TrainTokenizer.save_vocab_and_merges(output_path, vocab, merges)

    output_volume.commit()   # flush writes so they're visible outside the container

    print(f"      Saved to Modal Volume at {CONTAINER_OUTPUT_DIR}/")
    return {
        "vocab_tokens": len(vocab),
        "num_merges":   len(merges),
    }


# ---------------------------------------------------------------------------
# Local entrypoint — kicks off the remote job, then downloads results
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    import os

    print("=" * 50)
    print("CS336 BPE Tokenizer — Modal Training Run")
    print(f"  vocab_size  : {VOCAB_SIZE}")
    print(f"  RAM         : {MEMORY_GB} GB")
    print(f"  CPUs        : {CPU_COUNT}")
    print("=" * 50)

    result = train_tokenizer.remote()
    print(f"\nTraining result: {result}")

    # ------------------------------------------------------------------
    # Download outputs from the Modal Volume to ./bpe_outputs/ locally
    # ------------------------------------------------------------------
    local_out = "./bpe_outputs"
    os.makedirs(local_out, exist_ok=True)

    vol = modal.Volume.from_name("bpe-outputs")
    for filename in ["vocab.json", "merges.txt"]:
        local_path = os.path.join(local_out, filename)
        with open(local_path, "wb") as f:
            for chunk in vol.read_file(filename):
                f.write(chunk)
        size_kb = os.path.getsize(local_path) / 1024
        print(f"Downloaded → {local_path}  ({size_kb:.0f} KB)")

    print(f"\nDone. Outputs in {local_out}/")