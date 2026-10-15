"""Run the CS336 language-model trainer on Modal.

Usage:
    cd assignment1_basics

    # Upload tokenized data once:
    uv run modal volume put llm-data data/tinystories_train.npy tinystories_train.npy
    uv run modal volume put llm-data data/tinystories_valid.npy tinystories_valid.npy

    # Optional, for Weights & Biases logging:
    export WANDB_API_KEY=...

    # Start a remote GPU training run:
    uv run modal run cs336_basics/train_llm_modal.py --run-name modal-bs64-lr1e-3
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


APP_NAME = "cs336-llm-trainer"
GPU_TYPE = "A10G"
CPU_COUNT = 8
MEMORY_MB = 32 * 1024
TIMEOUT_SECONDS = 60 * 60 * 12

DATA_VOLUME_NAME = "llm-data"
OUTPUT_VOLUME_NAME = "llm-outputs"

CONTAINER_DATA_DIR = "/data"
CONTAINER_OUTPUT_DIR = "/outputs"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


app = modal.App(APP_NAME)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "einops>=0.8",
        "einx>=0.4",
        "jaxtyping>=0.3",
        "numpy>=2.4",
        "psutil>=7",
        "pyyaml>=6.0",
        "regex>=2026.3.32",
        "tiktoken>=0.12.0",
        "torch~=2.11.0",
        "tqdm>=4.67",
        "wandb>=0.25",
        "tensorboard>=2.20",
    )
    .add_local_dir(PROJECT_ROOT / "cs336_basics", remote_path="/app/cs336_basics")
    .add_local_dir(PROJECT_ROOT / "configs", remote_path="/app/configs")
)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_COUNT,
    memory=MEMORY_MB,
    timeout=TIMEOUT_SECONDS,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        CONTAINER_DATA_DIR: data_volume,
        CONTAINER_OUTPUT_DIR: output_volume,
    },
)
def train_llm(
    config: str,
    train_data: str,
    valid_data: str | None,
    run_name: str,
    vocab_size: int | None,
    batch_size: int | None,
    lr: float | None,
    max_steps: int | None,
    max_duration_minutes: float | None,
    log_interval: int,
    save_interval: int,
    val_check_interval: int,
    max_val_batches: int | None,
    tracking_backend: str,
    wandb_entity: str | None,
    auto_lr_find: bool,
    lr_find_min_lr: float | None,
    lr_find_max_lr: float | None,
    lr_find_steps: int | None,
    dtype: str,
    num_workers: int,
    resume_from: str | None,
) -> dict[str, str | int | None]:
    import os
    import subprocess
    import sys
    from pathlib import Path

    sys.path.insert(0, "/app")
    os.chdir("/app")

    checkpoint_dir = f"{CONTAINER_OUTPUT_DIR}/checkpoints/{run_name}"
    tensorboard_dir = f"{CONTAINER_OUTPUT_DIR}/runs"
    saved_config = f"{CONTAINER_OUTPUT_DIR}/configs/{run_name}.json"

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    Path(saved_config).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "cs336_basics.train_llm",
        "--config",
        config,
        "--train-data",
        train_data,
        "--run-name",
        run_name,
        "--device",
        "cuda",
        "--dtype",
        dtype,
        "--num-workers",
        str(num_workers),
        "--checkpoint-dir",
        checkpoint_dir,
        "--enable-tracking",
        "--tracking-backend",
        tracking_backend,
        "--tensorboard-dir",
        tensorboard_dir,
        "--save-config",
        saved_config,
        "--log-interval",
        str(log_interval),
        "--save-interval",
        str(save_interval),
        "--val-check-interval",
        str(val_check_interval),
    ]

    if auto_lr_find:
        cmd.append("--auto-lr-find")
    if lr_find_min_lr is not None:
        cmd.extend(["--lr-find-min-lr", str(lr_find_min_lr)])
    if lr_find_max_lr is not None:
        cmd.extend(["--lr-find-max-lr", str(lr_find_max_lr)])
    if lr_find_steps is not None:
        cmd.extend(["--lr-find-steps", str(lr_find_steps)])
    if valid_data:
        cmd.extend(["--valid-data", valid_data])
    if vocab_size is not None:
        cmd.extend(["--vocab-size", str(vocab_size)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if max_steps is not None:
        cmd.extend(["--max-steps", str(max_steps)])
    if max_duration_minutes is not None:
        cmd.extend(["--max-duration-minutes", str(max_duration_minutes)])
    if max_val_batches is not None:
        cmd.extend(["--max-val-batches", str(max_val_batches)])
    if wandb_entity:
        cmd.extend(["--wandb-entity", wandb_entity])
    if resume_from:
        cmd.extend(["--resume-from", resume_from])

    print("Running remote training command:")
    print(" ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = f"/app:{env.get('PYTHONPATH', '')}"
    subprocess.run(cmd, check=True, env=env)

    output_volume.commit()

    checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint-*.pt"))
    return {
        "run_name": run_name,
        "checkpoint_dir": checkpoint_dir,
        "num_checkpoints": len(checkpoints),
        "latest_checkpoint": str(checkpoints[-1]) if checkpoints else None,
        "saved_config": saved_config,
        "tensorboard_dir": tensorboard_dir,
    }


@app.local_entrypoint()
def main(
    config: str = "configs/default_config.yaml",
    train_data: str = f"{CONTAINER_DATA_DIR}/owt_train.npy",
    valid_data: str = f"{CONTAINER_DATA_DIR}/owt_valid.npy",
    run_name: str = "modal-run",
    vocab_size: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    max_steps: int | None = None,
    max_duration_minutes: float | None = 45.0,
    log_interval: int = 50,
    save_interval: int = 1000,
    val_check_interval: int = 0,
    max_val_batches: int | None = -1,
    tracking_backend: str = "wandb",
    wandb_entity: str | None = None,
    auto_lr_find: bool = False,
    lr_find_min_lr: float | None = None,
    lr_find_max_lr: float | None = None,
    lr_find_steps: int | None = None,
    dtype: str = "bfloat16",
    num_workers: int = 8,
    resume_from: str | None = None,
):
    print("=" * 60)
    print("CS336 LLM Modal Training")
    print(f"  app          : {APP_NAME}")
    print(f"  gpu          : {GPU_TYPE}")
    print(f"  data volume  : {DATA_VOLUME_NAME}")
    print(f"  output volume: {OUTPUT_VOLUME_NAME}")
    print(f"  run_name     : {run_name}")
    print("=" * 60)

    result = train_llm.remote(
        config=config,
        train_data=train_data,
        valid_data=valid_data,
        run_name=run_name,
        vocab_size=vocab_size,
        batch_size=batch_size,
        lr=lr,
        max_steps=max_steps,
        max_duration_minutes=max_duration_minutes,
        log_interval=log_interval,
        save_interval=save_interval,
        val_check_interval=val_check_interval,
        max_val_batches=max_val_batches,
        tracking_backend=tracking_backend,
        wandb_entity=wandb_entity,
        auto_lr_find=auto_lr_find,
        lr_find_min_lr=lr_find_min_lr,
        lr_find_max_lr=lr_find_max_lr,
        lr_find_steps=lr_find_steps,
        dtype=dtype,
        num_workers=num_workers,
        resume_from=resume_from,
    )

    print("\nRemote training finished.")
    for key, value in result.items():
        print(f"  {key}: {value}")
