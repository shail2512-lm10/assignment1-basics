#!/usr/bin/env bash
set -e

cd /Users/shailshah/CodeRepos/cs336/assignment1_basics

for bs in 8 16 32; do
  for lr in 6e-5 8e-5 1e-4 1.3e-4 1.6e-4; do
    run_name="bs${bs}_lr${lr}"

    echo
    echo "========================================"
    echo "Running: batch_size=$bs, lr=$lr, run_name=$run_name"
    echo "========================================"

    uv run python cs336_basics/train_llm.py \
      --config configs/default_config.yaml \
      --batch-size "$bs" \
      --lr "$lr" \
      --run-name "$run_name" \
      --max-steps 5000 \
      --log-interval 50 \
      --save-interval 500
  done
done