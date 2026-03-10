"""Experiment tracking module supporting wandb, mlflow, and tensorboard."""

import time
import logging
from pathlib import Path
from datetime import datetime

import torch.nn as nn


logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracking interface supporting multiple backends.
    
    Supports:
    - Weights & Biases (wandb)
    - MLflow
    - TensorBoard
    
    Tracks metrics with respect to both global steps and wallclock time.
    """
    
    def __init__(
        self,
        enabled: bool = False,
        backend: str = "wandb",
        project_name: str = "cs336-basics",
        experiment_name: str | None = None,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
        notes: str | None = None,
        wandb_entity: str | None = None,
        mlflow_tracking_uri: str | None = None,
        tensorboard_dir: str = "runs",
        log_interval: int = 10,
    ):
        """
        Initialize experiment tracker with specified backends.
        
        Args:
            enabled: Whether to enable experiment tracking
            backend: Comma-separated list of backends (wandb, mlflow, tensorboard)
            project_name: Project name for tracking
            experiment_name: Experiment name
            run_name: Run name for this training session
            tags: Dictionary of tags to track
            notes: Notes about the experiment
            wandb_entity: Weights & Biases entity (username or team)
            mlflow_tracking_uri: MLflow tracking server URI
            tensorboard_dir: Directory for tensorboard logs
            log_interval: Logging frequency in steps
        """
        self.enabled = enabled
        self.log_interval = log_interval
        self.start_time = time.time()
        
        self.callbacks = {}
        self.global_step = 0
        
        if not enabled:
            logger.info("Experiment tracking is disabled")
            return
        
        # Parse backends
        backends = [b.strip().lower() for b in backend.split(",")]
        
        # Initialize wandb
        if "wandb" in backends:
            self._init_wandb(project_name, experiment_name, run_name, tags, notes, wandb_entity)
        
        # Initialize mlflow
        if "mlflow" in backends:
            self._init_mlflow(project_name, experiment_name, run_name, tags, notes, mlflow_tracking_uri)
        
        # Initialize tensorboard
        if "tensorboard" in backends:
            self._init_tensorboard(tensorboard_dir, run_name)
        
        logger.info(f"Initialized tracker with backends: {', '.join([b for b in backends if b in self.callbacks])}")
    
    def _init_wandb(
        self,
        project_name: str,
        experiment_name: str | None,
        run_name: str | None,
        tags: dict[str, str] | None,
        notes: str | None,
        entity: str | None,
    ) -> None:
        """Initialize Weights & Biases tracking."""
        try:
            import wandb
            
            config = {
                "project": project_name,
            }
            
            if run_name:
                config["name"] = run_name
            if entity:
                config["entity"] = entity
            if tags:
                config["tags"] = list(tags.keys())
            if notes:
                config["notes"] = notes
            
            wandb.init(**config)
            
            if tags:
                wandb.config.update(tags)
            
            self.callbacks["wandb"] = WandBCallback()
            logger.info("Initialized Weights & Biases tracking")
        except ImportError:
            logger.warning("wandb not installed, skipping wandb initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    def _init_mlflow(
        self,
        project_name: str,
        experiment_name: str | None,
        run_name: str | None,
        tags: dict[str, str] | None,
        notes: str | None,
        tracking_uri: str | None,
    ) -> None:
        """Initialize MLflow tracking."""
        try:
            import mlflow
            
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            mlflow.set_experiment(experiment_name or project_name)
            
            mlflow.start_run(run_name=run_name)
            
            if tags:
                mlflow.set_tags(tags)
            
            if notes:
                mlflow.set_tag("notes", notes)
            
            self.callbacks["mlflow"] = MLflowCallback()
            logger.info("Initialized MLflow tracking")
        except ImportError:
            logger.warning("mlflow not installed, skipping mlflow initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize mlflow: {e}")
    
    def _init_tensorboard(
        self,
        tensorboard_dir: str,
        run_name: str | None,
    ) -> None:
        """Initialize TensorBoard tracking."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            
            # Create run name from timestamp if not provided
            if not run_name:
                run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            log_dir = Path(tensorboard_dir) / run_name
            log_dir.mkdir(parents=True, exist_ok=True)
            
            writer = SummaryWriter(str(log_dir))
            self.callbacks["tensorboard"] = TensorBoardCallback(writer)
            logger.info(f"Initialized TensorBoard tracking at {log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed, skipping tensorboard initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize tensorboard: {e}")
    
    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int,
        wallclock_time: float | None = None,
    ) -> None:
        """
        Log metrics to all configured backends.
        
        Args:
            metrics: Dictionary of metric name -> value
            step: Current global step (number of gradient updates)
            wallclock_time: Elapsed time in seconds (auto-calculated if not provided)
        """
        if not self.enabled or step % self.log_interval != 0:
            return
        
        if wallclock_time is None:
            wallclock_time = time.time() - self.start_time
        
        self.global_step = step
        
        # Add wallclock time to metrics
        metrics_with_time = {
            **metrics,
            "wallclock_time_seconds": wallclock_time,
            "wallclock_time_hours": wallclock_time / 3600,
        }
        
        for backend_name, callback in self.callbacks.items():
            try:
                callback.log_metrics(metrics_with_time, step)
            except Exception as e:
                logger.warning(f"Failed to log metrics to {backend_name}: {e}")
    
    def log_model_checkpoint(
        self,
        model: nn.Module,
        checkpoint_path: str | Path,
        step: int,
        is_best: bool = False,
    ) -> None:
        """
        Log model checkpoint artifacts.
        
        Args:
            model: PyTorch model
            checkpoint_path: Path to checkpoint file
            step: Current global step
            is_best: Whether this is the best model so far
        """
        if not self.enabled:
            return
        
        for backend_name, callback in self.callbacks.items():
            try:
                callback.log_model_checkpoint(checkpoint_path, step, is_best)
            except Exception as e:
                logger.warning(f"Failed to log checkpoint to {backend_name}: {e}")
    
    def log_config(self, config: dict[str, any]) -> None:
        """
        Log training configuration.
        
        Args:
            config: Configuration dictionary
        """
        if not self.enabled:
            return
        
        for backend_name, callback in self.callbacks.items():
            try:
                callback.log_config(config)
            except Exception as e:
                logger.warning(f"Failed to log config to {backend_name}: {e}")
    
    def log_gradient_stats(
        self,
        model: nn.Module,
        step: int,
    ) -> None:
        """
        Log gradient statistics (norms, magnitudes, etc.).
        
        Args:
            model: PyTorch model
            step: Current global step
        """
        if not self.enabled or step % self.log_interval != 0:
            return
        
        # Compute gradient statistics
        total_norm = 0.0
        param_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_norms[f"grad_norm_{name}"] = param_norm.item()
        
        total_norm = total_norm ** 0.5
        grad_stats = {
            "gradient_norm_total": total_norm,
            **param_norms,
        }
        
        for backend_name, callback in self.callbacks.items():
            try:
                callback.log_metrics(grad_stats, step)
            except Exception as e:
                logger.warning(f"Failed to log gradient stats to {backend_name}: {e}")
    
    def finish(self) -> None:
        """Finish tracking and cleanup."""
        if not self.enabled:
            return
        
        for backend_name, callback in self.callbacks.items():
            try:
                callback.finish()
            except Exception as e:
                logger.warning(f"Failed to finish {backend_name}: {e}")
        
        logger.info("Experiment tracking finished")


class WandBCallback:
    """Callback for Weights & Biases."""
    
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to wandb."""
        import wandb
        wandb.log(metrics, step=step)
    
    def log_model_checkpoint(self, checkpoint_path: str | Path, step: int, is_best: bool = False) -> None:
        """Log model checkpoint to wandb."""
        import wandb
        
        artifact_name = f"model-checkpoint-{'best' if is_best else f'step{step}'}"
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
    
    def log_config(self, config: dict[str, any]) -> None:
        """Log configuration to wandb."""
        import wandb
        wandb.config.update(config)
    
    def finish(self) -> None:
        """Finish wandb run."""
        import wandb
        wandb.finish()


class MLflowCallback:
    """Callback for MLflow."""
    
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to mlflow."""
        import mlflow
        
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow.log_metric(metric_name, metric_value, step=step)
    
    def log_model_checkpoint(self, checkpoint_path: str | Path, step: int, is_best: bool = False) -> None:
        """Log model checkpoint to mlflow."""
        import mlflow
        
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    
    def log_config(self, config: dict[str, any]) -> None:
        """Log configuration to mlflow."""
        import mlflow
        
        # Flatten config dictionary for mlflow
        def flatten_dict(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flat_config = flatten_dict(config)
        
        for key, value in flat_config.items():
            try:
                if isinstance(value, (int, float, str, bool)):
                    mlflow.log_param(key, value)
            except Exception:
                pass  # Skip non-loggable values
    
    def finish(self) -> None:
        """Finish mlflow run."""
        import mlflow
        mlflow.end_run()


class TensorBoardCallback:
    """Callback for TensorBoard."""
    
    def __init__(self, writer):
        """Initialize with a SummaryWriter."""
        self.writer = writer
    
    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log metrics to tensorboard."""
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.writer.add_scalar(metric_name, metric_value, global_step=step)
        self.writer.flush()
    
    def log_model_checkpoint(self, checkpoint_path: str | Path, step: int, is_best: bool = False) -> None:
        """Log model checkpoint info to tensorboard."""
        tag = f"checkpoint/{'best' if is_best else f'step{step}'}"
        self.writer.add_text(tag, str(checkpoint_path), global_step=step)
    
    def log_config(self, config: dict[str, any]) -> None:
        """Log configuration to tensorboard."""
        from tensorboard.plugins.hparams import api as hp
        
        # Convert config to hparams format
        hparams = {}
        
        def flatten_dict(d, parent_key="", sep="_"):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items
        
        hparams = flatten_dict(config)
        
        # Filter to only simple types that tensorboard can log
        simple_hparams = {}
        for k, v in hparams.items():
            if isinstance(v, (int, float, str, bool)):
                simple_hparams[k] = v
        
        if simple_hparams:
            try:
                hp.hparams(simple_hparams)
            except Exception:
                # Fallback: just log as text
                import json
                self.writer.add_text("hparams", json.dumps(simple_hparams, indent=2))
    
    def finish(self) -> None:
        """Finish tensorboard logging."""
        self.writer.close()
