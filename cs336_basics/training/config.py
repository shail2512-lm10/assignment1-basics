"""Configuration management for training hyperparameters and settings."""

import json
import yaml
from dataclasses import dataclass, field, asdict
from pathlib import Path
import torch


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    name: str = "adamw"  # adamw, sgd, adam
    lr: float = 1e-4
    betas: tuple = (0.9, 0.999)
    weight_decay: float = 0.01
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    name: str = "cosine"  # cosine, constant, linear, exponential
    lr_max: float = 1e-4
    lr_min: float = 0.0
    warmup_steps: int = 1000
    total_steps: int = 10000


@dataclass
class DataConfig:
    """Configuration for data loading."""
    train_data_path: str = "data/owt_train.txt"
    valid_data_path: str | None = None
    batch_size: int = 32
    context_length: int = 512
    num_workers: int = 0
    pin_memory: bool = False
    shuffle: bool = True


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    vocab_size: int = 50257
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = 3072
    context_length: int | None = 1024
    theta: int | None = 10000
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training loop."""
    max_epochs: int = 10
    max_steps: int | None = None  # If set, overrides max_epochs
    gradient_accumulation_steps: int = 1
    gradient_clip_val: float | None = 1.0
    val_check_interval: int = 100  # Check validation every N steps
    save_interval: int = 500
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"  # float32, float16, bfloat16
    seed: int = 42


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing."""
    checkpoint_dir: str = "checkpoints"
    save_best_only: bool = False
    resume_from_checkpoint: str | None = None
    keep_last_n: int = 3  # Keep only last N checkpoints


@dataclass
class ExperimentTrackingConfig:
    """Configuration for experiment tracking (wandb, mlflow, tensorboard)."""
    enabled: bool = False
    backend: str = "wandb"  # wandb, mlflow, tensorboard, or multiple separated by commas
    project_name: str = "cs336-basics"
    experiment_name: str | None = None
    run_name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    notes: str | None = None
    
    # Tracking frequency
    log_interval: int = 10  # Log metrics every N steps
    log_model: bool = False  # Log model artifacts
    log_gradients: bool = False  # Log gradient statistics
    
    # Backends specific
    wandb_entity: str | None = None  # wandb username/team
    mlflow_tracking_uri: str | None = None  # mlflow server URI
    tensorboard_dir: str = "runs"  # Directory for tensorboard logs


@dataclass
class TrainerConfig:
    """Main configuration class combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    experiment_tracking: ExperimentTrackingConfig = field(default_factory=ExperimentTrackingConfig)
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, any]) -> "TrainerConfig":
        """Create TrainerConfig from a dictionary."""
        config = cls()
        
        if "model" in config_dict:
            config.model = ModelConfig(**config_dict["model"])
        if "data" in config_dict:
            config.data = DataConfig(**config_dict["data"])
        if "optimizer" in config_dict:
            config.optimizer = OptimizerConfig(**config_dict["optimizer"])
        if "scheduler" in config_dict:
            config.scheduler = SchedulerConfig(**config_dict["scheduler"])
        if "training" in config_dict:
            config.training = TrainingConfig(**config_dict["training"])
        if "checkpoint" in config_dict:
            config.checkpoint = CheckpointConfig(**config_dict["checkpoint"])
        if "experiment_tracking" in config_dict:
            config.experiment_tracking = ExperimentTrackingConfig(**config_dict["experiment_tracking"])
        
        return config
    
    @classmethod
    def from_json(cls, filepath: str | Path) -> "TrainerConfig":
        """Load configuration from JSON file."""
        with open(filepath) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_yaml(cls, filepath: str | Path) -> "TrainerConfig":
        """Load configuration from YAML file."""
        with open(filepath) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> dict[str, any]:
        """Convert config to dictionary."""
        data = asdict(self)
        # Convert betas tuple to list for JSON serialization
        if isinstance(data["optimizer"]["betas"], tuple):
            data["optimizer"]["betas"] = list(data["optimizer"]["betas"])
        return data
    
    def to_json(self, filepath: str | Path, indent: int = 2) -> None:
        """Save configuration to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)
    
    def to_yaml(self, filepath: str | Path) -> None:
        """Save configuration to YAML file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __repr__(self) -> str:
        """Pretty string representation."""
        lines = ["TrainerConfig("]
        lines.append(f"  model={self.model}")
        lines.append(f"  data={self.data}")
        lines.append(f"  optimizer={self.optimizer}")
        lines.append(f"  scheduler={self.scheduler}")
        lines.append(f"  training={self.training}")
        lines.append(f"  checkpoint={self.checkpoint}")
        lines.append(f"  experiment_tracking={self.experiment_tracking}")
        lines.append(")")
        return "\n".join(lines)
