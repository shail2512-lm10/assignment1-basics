#!/usr/bin/env python3
"""
Training script for LLM (Language Model) using Transformer architecture.

Supports:
- Loading configuration from JSON/YAML files
- Overriding config parameters via command-line arguments
- Training with automatic checkpoint management
- Experiment tracking integration

Usage:
    # Train with default config
    python train_llm.py
    
    # Train with custom config file
    python train_llm.py --config configs/custom_config.json
    
    # Override specific parameters
    python train_llm.py --batch-size 16 --max-steps 5000 --lr 5e-5
    
    # Resume from checkpoint
    python train_llm.py --resume-from checkpoints/checkpoint_epoch_2.pt
    
    # Use YAML config with overrides
    python train_llm.py --config configs/default_config.yaml --device cpu --max-epochs 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from cs336_basics.training import Trainer, TrainerConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train an LLM using Transformer architecture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_config.json",
        help="Path to configuration file (JSON or YAML)",
    )
    
    # Model parameters
    parser.add_argument(
        "--vocab-size",
        type=int,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        help="Model dimension (embedding size)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="Number of transformer blocks",
    )
    parser.add_argument(
        "--d-ff",
        type=int,
        help="Feed-forward network dimension",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout probability",
    )
    
    # Data parameters
    parser.add_argument(
        "--train-data",
        type=str,
        help="Path to training data",
    )
    parser.add_argument(
        "--valid-data",
        type=str,
        help="Path to validation data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Context/sequence length for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of data loading workers",
    )
    
    # Optimizer parameters
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adamw", "adam", "sgd"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        help="Weight decay",
    )
    
    # Scheduler parameters
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "constant", "linear", "exponential"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Number of warmup steps",
    )
    
    # Training parameters
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum number of training steps (overrides epochs if set)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        help="Logging interval (steps)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        help="Checkpoint saving interval (steps)",
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        help="Validation check interval (steps)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient-clip-val",
        type=float,
        help="Gradient clipping value",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Data type for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
    )
    
    # Checkpoint parameters
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        help="Save only the best model based on validation loss",
    )
    
    # Experiment tracking
    parser.add_argument(
        "--enable-tracking",
        action="store_true",
        help="Enable experiment tracking (wandb/mlflow/tensorboard)",
    )
    parser.add_argument(
        "--tracking-backend",
        type=str,
        choices=["wandb", "mlflow", "tensorboard"],
        help="Experiment tracking backend",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        help="Experiment tracking project name",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        help="Run name for tracking",
    )
    parser.add_argument(
        "--tracking-log-interval",
        type=int,
        help="Experiment tracking logging interval (steps)",
    )
    parser.add_argument(
        "--log-model",
        action="store_true",
        help="Log model artifacts to tracking system",
    )
    parser.add_argument(
        "--log-gradients",
        action="store_true",
        help="Log gradient statistics to tracking system",
    )
    parser.add_argument(
        "--tracking-notes",
        type=str,
        help="Notes for the experiment tracking run",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="*",
        help="Tags for experiment tracking (format: key1=value1 key2=value2)",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        help="Weights & Biases entity (username/team)",
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        help="MLflow tracking server URI",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        help="Directory for TensorBoard logs",
    )
    
    # Utility arguments
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load config and print it without training",
    )
    parser.add_argument(
        "--save-config",
        type=str,
        help="Save the final configuration (after overrides) to a file",
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> TrainerConfig:
    """
    Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to config file (JSON or YAML)
        
    Returns:
        TrainerConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is not recognized
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    logger.info(f"Loading configuration from {config_path}")
    
    suffix = config_path.suffix.lower()
    
    if suffix == ".json":
        with open(config_path) as f:
            config = TrainerConfig.from_json(f)
    elif suffix in [".yaml", ".yml"]:
        with open(config_path) as f:
            config = TrainerConfig.from_yaml(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}")
    
    return config


def apply_overrides(config: TrainerConfig, args: argparse.Namespace) -> None:
    """
    Apply command-line argument overrides to the configuration.
    
    Args:
        config: TrainerConfig instance to modify
        args: Parsed command-line arguments
    """
    # Model overrides
    if args.vocab_size is not None:
        config.model.vocab_size = args.vocab_size
    if args.d_model is not None:
        config.model.d_model = args.d_model
    if args.num_heads is not None:
        config.model.num_heads = args.num_heads
    if args.num_layers is not None:
        config.model.num_layers = args.num_layers
    if args.d_ff is not None:
        config.model.d_ff = args.d_ff
    if args.context_length is not None:
        config.model.context_length = args.context_length
    if args.dropout is not None:
        config.model.dropout = args.dropout
    
    # Data overrides
    if args.train_data is not None:
        config.data.train_data_path = args.train_data
    if args.valid_data is not None:
        config.data.valid_data_path = args.valid_data
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.context_length is not None:
        config.data.context_length = args.context_length
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    
    # Optimizer overrides
    if args.optimizer is not None:
        config.optimizer.name = args.optimizer
    if args.lr is not None:
        config.optimizer.lr = args.lr
    if args.weight_decay is not None:
        config.optimizer.weight_decay = args.weight_decay
    
    # Scheduler overrides
    if args.scheduler is not None:
        config.scheduler.name = args.scheduler
    if args.warmup_steps is not None:
        config.scheduler.warmup_steps = args.warmup_steps
    
    # Training overrides
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.log_interval is not None:
        config.training.log_interval = args.log_interval
    if args.save_interval is not None:
        config.training.save_interval = args.save_interval
    if args.val_check_interval is not None:
        config.training.val_check_interval = args.val_check_interval
    if args.gradient_accumulation_steps is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.gradient_clip_val is not None:
        config.training.gradient_clip_val = args.gradient_clip_val
    if args.device is not None:
        config.training.device = args.device
    if args.dtype is not None:
        config.training.dtype = args.dtype
    if args.seed is not None:
        config.training.seed = args.seed
    
    # Checkpoint overrides
    if args.checkpoint_dir is not None:
        config.checkpoint.checkpoint_dir = args.checkpoint_dir
    if args.resume_from is not None:
        config.checkpoint.resume_from_checkpoint = args.resume_from
    if args.save_best_only:
        config.checkpoint.save_best_only = True
    
    # Experiment tracking overrides
    if args.enable_tracking:
        config.experiment_tracking.enabled = True
    if args.tracking_backend is not None:
        config.experiment_tracking.backend = args.tracking_backend
    if args.project_name is not None:
        config.experiment_tracking.project_name = args.project_name
    if args.experiment_name is not None:
        config.experiment_tracking.experiment_name = args.experiment_name
    if args.run_name is not None:
        config.experiment_tracking.run_name = args.run_name
    if args.tracking_log_interval is not None:
        config.experiment_tracking.log_interval = args.tracking_log_interval
    if args.log_model:
        config.experiment_tracking.log_model = True
    if args.log_gradients:
        config.experiment_tracking.log_gradients = True
    if args.tracking_notes is not None:
        config.experiment_tracking.notes = args.tracking_notes
    if args.wandb_entity is not None:
        config.experiment_tracking.wandb_entity = args.wandb_entity
    if args.mlflow_tracking_uri is not None:
        config.experiment_tracking.mlflow_tracking_uri = args.mlflow_tracking_uri
    if args.tensorboard_dir is not None:
        config.experiment_tracking.tensorboard_dir = args.tensorboard_dir
    
    # Parse tags from command-line format (key1=value1 key2=value2)
    if args.tags:
        tags_dict = {}
        for tag in args.tags:
            if "=" in tag:
                key, value = tag.split("=", 1)
                tags_dict[key] = value
            else:
                logger.warning(f"Skipping invalid tag format: {tag} (expected key=value)")
        if tags_dict:
            config.experiment_tracking.tags = tags_dict


def save_config_to_file(config: TrainerConfig, filepath: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: TrainerConfig instance to save
        filepath: Path to save the config to
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary for JSON serialization
    config_dict = config.to_dict()
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    logger.info(f"Configuration saved to {filepath}")


def print_config(config: TrainerConfig) -> None:
    """Print configuration in a readable format."""
    config_dict = config.to_dict()
    
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(json.dumps(config_dict, indent=2))
    print("=" * 50 + "\n")


def main():
    """Main entry point for the training script."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Apply command-line overrides
        apply_overrides(config, args)
        
        # Print configuration
        print_config(config)
        
        # Save configuration if requested
        if args.save_config:
            save_config_to_file(config, args.save_config)
        
        # Dry run: just print config and exit
        if args.dry_run:
            logger.info("Dry run: configuration loaded successfully, exiting")
            return 0
        
        # Set random seed
        torch.manual_seed(config.training.seed)
        logger.info(f"Random seed set to {config.training.seed}")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(config=config)
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Print summary
        summary = trainer.get_training_summary()
        print("\n" + "=" * 50)
        print("TRAINING SUMMARY")
        print("=" * 50)
        print(json.dumps(summary, indent=2, default=str))
        print("=" * 50)
        
        logger.info("Training completed successfully")
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.exception(f"Training failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
