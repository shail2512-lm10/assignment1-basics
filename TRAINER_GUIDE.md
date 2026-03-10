# PyTorch Lightning-Style Trainer Guide

A comprehensive guide for using the PyTorch Lightning-style Trainer for training transformer models with configuration file support.

## Overview

The `Trainer` class provides a high-level interface for managing the complete training pipeline:

- **Automatic Setup**: Device management, dtype conversion, model/optimizer/dataloader initialization
- **Configuration-Based**: Load and manage hyperparameters from JSON/YAML files
- **Training Features**: Progress tracking, learning rate scheduling, gradient clipping, validation
- **Checkpointing**: Automatic checkpoint saving/loading with cleanup
- **Logging**: Comprehensive logging of training metrics

## Configuration System

### Configuration Classes

The configuration system uses dataclasses organized hierarchically:

1. **TrainerConfig** - Main configuration container
2. **ModelConfig** - Architecture settings (vocab_size, d_model, num_heads, etc.)
3. **DataConfig** - Data loading settings (paths, batch_size, context_length)
4. **OptimizerConfig** - Optimizer parameters (lr, weight_decay, betas)
5. **SchedulerConfig** - Learning rate scheduler settings
6. **TrainingConfig** - Training loop parameters (epochs, gradient clipping, device)
7. **CheckpointConfig** - Checkpoint management settings

### Loading Configurations

#### From JSON File
```python
from cs336_basics.training import TrainerConfig

config = TrainerConfig.from_json("configs/default_config.json")
```

#### From YAML File
```python
config = TrainerConfig.from_yaml("configs/default_config.yaml")
```

#### From Dictionary
```python
config_dict = {
    "model": {"vocab_size": 50257, "d_model": 768, ...},
    "data": {"batch_size": 32, ...},
    # ... other configs
}
config = TrainerConfig.from_dict(config_dict)
```

#### Programmatic Creation
```python
from cs336_basics.training import TrainerConfig, ModelConfig, DataConfig

config = TrainerConfig(
    model=ModelConfig(vocab_size=50257, d_model=768, num_heads=12, ...),
    data=DataConfig(batch_size=32, context_length=512, ...),
)
```

### Modifying Configurations

```python
config = TrainerConfig.from_json("configs/default_config.json")

# Modify specific parameters
config.training.max_steps = 5000
config.optimizer.lr = 1e-4
config.training.gradient_clip_val = 1.0

# Create trainer with modified config
trainer = Trainer(config=config)
```

### Saving Configurations

```python
# Save as JSON
config.to_json("my_config.json")

# Save as YAML
config.to_yaml("my_config.yaml")
```

## Training

### Basic Training

```python
from cs336_basics.training import Trainer, TrainerConfig

# Load configuration
config = TrainerConfig.from_json("configs/default_config.json")

# Create trainer (automatically initializes model, optimizer, dataloaders)
trainer = Trainer(config=config)

# Start training
trainer.train()

# Get training summary
summary = trainer.get_training_summary()
print(f"Total steps: {summary['total_steps']}")
print(f"Best validation loss: {summary['best_val_loss']}")
```

### Custom Training with Manual Components

```python
from cs336_basics.training import Trainer, TrainerConfig
from cs336_basics.transformer import TransformerLM

# Create components manually
config = TrainerConfig.from_json("configs/default_config.json")
model = TransformerLM(d_model=768, num_heads=12, ...)
train_dataloader = ...  # Your custom dataloader
val_dataloader = ...    # Your custom dataloader

# Pass to trainer
trainer = Trainer(
    model=model,
    config=config,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
)

trainer.train()
```

### Resume from Checkpoint

```python
trainer = Trainer(config=config)

# Resume from specific checkpoint
trainer.train(resume_from_checkpoint="checkpoints/checkpoint-epoch2-step1500.pt")
```

Or set it in config:
```python
config.checkpoint.resume_from_checkpoint = "checkpoints/checkpoint-epoch2-step1500.pt"
trainer = Trainer(config=config)
trainer.train()
```

## Configuration Parameters Reference

### ModelConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| vocab_size | int | 50257 | Vocabulary size |
| d_model | int | 768 | Model dimension |
| num_heads | int | 12 | Number of attention heads |
| num_layers | int | 12 | Number of transformer layers |
| d_ff | int | 3072 | Feedforward dimension |
| max_seq_len | int | 1024 | Maximum sequence length |
| theta | int | 10000 | RoPE theta parameter |
| dropout | float | 0.1 | Dropout rate |

### DataConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| train_data_path | str | "data/owt_train.txt" | Path to training data |
| valid_data_path | str | null | Path to validation data (optional) |
| batch_size | int | 32 | Batch size |
| context_length | int | 512 | Sequence context length |
| num_workers | int | 0 | DataLoader workers |
| pin_memory | bool | false | Pin memory for faster GPU transfer |
| shuffle | bool | true | Shuffle training batches |

### OptimizerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | "adamw" | Optimizer: "adamw", "adam", "sgd" |
| lr | float | 1e-4 | Learning rate |
| betas | tuple | (0.9, 0.999) | Adam betas or SGD momentum |
| weight_decay | float | 0.01 | Weight decay |
| eps | float | 1e-8 | Epsilon for numerical stability |

### SchedulerConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | "cosine" | Scheduler: "cosine", "constant", "linear", "exponential" |
| lr_max | float | 1e-4 | Maximum learning rate |
| lr_min | float | 0.0 | Minimum learning rate |
| warmup_steps | int | 1000 | Warmup steps |
| total_steps | int | 10000 | Total training steps |

### TrainingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_epochs | int | 10 | Maximum epochs (overridden by max_steps) |
| max_steps | int | null | Max training steps (overrides max_epochs) |
| gradient_accumulation_steps | int | 1 | Gradient accumulation steps |
| gradient_clip_val | float | 1.0 | Max gradient norm (null to disable) |
| val_check_interval | int | 100 | Validation check every N steps |
| save_interval | int | 500 | Save checkpoint every N steps |
| log_interval | int | 10 | Log metrics every N steps |
| device | str | "cuda" | Device: "cuda", "cpu" |
| dtype | str | "float32" | Data type: "float32", "float16", "bfloat16" |
| seed | int | 42 | Random seed |

### CheckpointConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| checkpoint_dir | str | "checkpoints" | Directory for saving checkpoints |
| save_best_only | bool | false | Only save best validation checkpoint |
| resume_from_checkpoint | str | null | Checkpoint to resume from |
| keep_last_n | int | 3 | Keep last N checkpoints |

### ExperimentTrackingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| enabled | bool | false | Enable experiment tracking |
| backend | str | "wandb" | Tracking backend(s); e.g. "wandb", "mlflow", "tensorboard" or multiple separated by commas |
| project_name | str | "cs336-basics" | Project name for the tracking service |
| experiment_name | str | null | Experiment name (optional) |
| run_name | str | null | Run name (optional) |
| tags | dict | {} | Tags to attach to the run |
| notes | str | null | Notes for the run |
| log_interval | int | 10 | Log metrics every N steps |
| log_model | bool | false | Log model artifacts |
| log_gradients | bool | false | Log gradient statistics |
| wandb_entity | str | null | WandB entity (user/team) |
| mlflow_tracking_uri | str | null | MLflow tracking server URI |
| tensorboard_dir | str | "runs" | Directory for TensorBoard logs |

## Features in Detail

### Learning Rate Scheduling

The trainer supports multiple scheduling strategies:

- **cosine**: Cosine annealing with warmup (default)
- **constant**: Constant learning rate
- **linear**: Linear warmup then linear decay
- **exponential**: Exponential decay

```python
config.scheduler.name = "cosine"
config.scheduler.lr_max = 1e-4
config.scheduler.warmup_steps = 1000
```

### Gradient Clipping

```python
# Enable gradient clipping
config.training.gradient_clip_val = 1.0

# Disable gradient clipping
config.training.gradient_clip_val = None
```

### Gradient Accumulation

```python
# Accumulate gradients over 4 batches
config.training.gradient_accumulation_steps = 4
```

Effective batch size = batch_size × gradient_accumulation_steps

### Checkpointing

```python
# Save checkpoint every N steps
config.checkpoint.save_interval = 500

# Only save best validation checkpoint
config.checkpoint.save_best_only = True

# Keep only last 5 checkpoints
config.checkpoint.keep_last_n = 5

# Automatically resume from last checkpoint
config.checkpoint.resume_from_checkpoint = "checkpoints/checkpoint-epoch2-step1000.pt"
```

### Validation

```python
# Enable validation
config.data.valid_data_path = "data/valid.txt"

# Check validation every N steps
config.training.val_check_interval = 100
```

### Logging

The trainer uses Python's logging module. Configure logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

Training metrics are logged at intervals:
- Training loss
- Current learning rate
- Training throughput (steps/sec)
- Validation loss
- Checkpoint saves

## Advanced Usage

### Custom Metrics

The trainer tracks:
- `train_losses`: List of training losses
- `val_losses`: List of validation losses
- `best_val_loss`: Best validation loss seen
- `global_step`: Total training steps completed
- `current_epoch`: Current epoch number

```python
trainer = Trainer(config=config)
trainer.train()

print(trainer.train_losses)
print(trainer.best_val_loss)
print(f"Completed {trainer.global_step} steps in {trainer.current_epoch} epochs")
```

### Training Summary

```python
summary = trainer.get_training_summary()
```

Returns:
- `total_steps`: Total steps completed
- `total_epochs`: Total epochs completed
- `train_losses`: List of training losses
- `val_losses`: List of validation losses
- `best_val_loss`: Best validation loss
- `model_params`: Total model parameters
- `device`: Device used
- `dtype`: Data type used

## Example Configs

### Small Model (Fast Testing)
```json
{
  "model": {
    "vocab_size": 50257,
    "d_model": 128,
    "num_heads": 4,
    "num_layers": 2,
    "d_ff": 512
  },
  "training": {
    "max_steps": 1000
  }
}
```

### Large Model (Production)
```json
{
  "model": {
    "vocab_size": 50257,
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 24,
    "d_ff": 4096
  },
  "training": {
    "max_epochs": 3
  }
}
```

### Multi-GPU (Future Enhancement)
```json
{
  "training": {
    "device": "cuda"
  }
}
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size`
- Reduce `context_length`
- Increase `gradient_accumulation_steps` for effective batch size
- Use `float16` or `bfloat16` dtype

### Slow Training
- Increase `num_workers` in DataConfig
- Set `pin_memory: true`
- Use `batch_size` that divides evenly into GPU memory

### NaN Loss
- Enable/increase gradient clipping
- Reduce learning rate
- Check data format (should be token IDs)

## Example Scripts

See `train_example.py` for complete working examples:

```bash
# Train with JSON config
python train_example.py --example json

# Train with YAML config
python train_example.py --example yaml

# Train with modified config
python train_example.py --example modify

# Train with programmatic config
python train_example.py --example programmatic

# Resume from checkpoint
python train_example.py --example resume
```

## API Reference

### Trainer Class

```python
class Trainer:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        config: Optional[TrainerConfig] = None,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloader: Optional[DataLoader] = None,
    )
    
    def train(self, resume_from_checkpoint: Optional[str | Path] = None) -> None
    def save_checkpoint(self, suffix: str = "") -> Path
    def load_checkpoint(self, checkpoint_path: str | Path) -> None
    def get_training_summary(self) -> Dict[str, Any]
```

### Properties

- `model`: PyTorch model
- `optimizer`: PyTorch optimizer
- `train_dataloader`: Training DataLoader
- `val_dataloader`: Validation DataLoader
- `config`: TrainerConfig instance
- `device`: PyTorch device
- `torch_dtype`: PyTorch dtype
- `global_step`: Current training step
- `current_epoch`: Current epoch
- `train_losses`: List of training losses
- `val_losses`: List of validation losses
- `best_val_loss`: Best validation loss
