"""PyTorch Lightning-style Trainer for managing the full training pipeline."""

import copy
import math
import time
from itertools import islice
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from einops import repeat

from cs336_basics.transformer import TransformerLM
from cs336_basics.training.config import TrainerConfig
from cs336_basics.training.loss import cross_entropy_loss
from cs336_basics.training.optimizer import AdamW, cosine_lr_schedule, gradient_clipping
from cs336_basics.training.data import get_batch_dataloader
from cs336_basics.training.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.training.experiment_tracker import ExperimentTracker


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Trainer:
    
    def __init__(
        self,
        model: nn.Module | None = None,
        config: "TrainerConfig | None" = None,
        train_dataloader: DataLoader | None = None,
        val_dataloader: DataLoader | None = None,
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: PyTorch model (if None, will be created from config)
            config: TrainerConfig instance (uses defaults if None)
            train_dataloader: Training DataLoader (created from config if None)
            val_dataloader: Validation DataLoader (optional)
        """
        self.config = config or TrainerConfig()
        
        # Setup device and dtype
        self._setup_device_and_dtype()
        
        # Initialize model
        if model is None:
            self.model = self._create_model_from_config()
        else:
            self.model = model

        self.model = self.model.to(self.device).to(self.torch_dtype)

        if self.device.type == "mps":
            self.model = torch.compile(self.model, backend="aot_eager")
        else:
            self.model = torch.compile(self.model)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize dataloaders
        if train_dataloader is None:
            self.train_dataloader = self._create_train_dataloader()
        else:
            self.train_dataloader = train_dataloader
        
        if val_dataloader is None and self.config.data.valid_data_path:
            self.val_dataloader = self._create_val_dataloader()
        else:
            self.val_dataloader = val_dataloader
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracker
        self.tracker = ExperimentTracker(
            enabled=self.config.experiment_tracking.enabled,
            backend=self.config.experiment_tracking.backend,
            project_name=self.config.experiment_tracking.project_name,
            experiment_name=self.config.experiment_tracking.experiment_name,
            run_name=self.config.experiment_tracking.run_name,
            tags=self.config.experiment_tracking.tags,
            notes=self.config.experiment_tracking.notes,
            wandb_entity=self.config.experiment_tracking.wandb_entity,
            mlflow_tracking_uri=self.config.experiment_tracking.mlflow_tracking_uri,
            tensorboard_dir=self.config.experiment_tracking.tensorboard_dir,
            log_interval=self.config.experiment_tracking.log_interval,
        )
        
        logger.info(f"Trainer initialized on device: {self.device} with dtype: {self.torch_dtype}")
    
    def _setup_device_and_dtype(self) -> None:
        """Setup device and PyTorch dtype."""
        # Device
        if self.config.training.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, using CPU")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(self.config.training.device)
        
        # Data type
        dtype_str = self.config.training.dtype.lower()
        if dtype_str == "float32":
            self.torch_dtype = torch.float32
        elif dtype_str == "float16":
            self.torch_dtype = torch.float16
        elif dtype_str == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            logger.warning(f"Unknown dtype {dtype_str}, using float32")
            self.torch_dtype = torch.float32
    
    def _create_model_from_config(self) -> nn.Module:
        """Create a TransformerLM model from config."""
        cfg = self.config.model
        logger.info(f"Creating TransformerLM with config: {cfg}")
        
        model = TransformerLM(
            vocab_size=cfg.vocab_size,
            context_length=cfg.context_length,
            num_layers=cfg.num_layers,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            theta=cfg.theta,
            device=self.device,
            dtype=self.torch_dtype
        )
        
        # Log parameter count
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {param_count:,} parameters")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        cfg = self.config.optimizer
        logger.info(f"Creating optimizer: {cfg.name} with lr={cfg.lr}")
        
        if cfg.name.lower() == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
                eps=cfg.eps
            )
        elif cfg.name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.lr,
                betas=cfg.betas,
                weight_decay=cfg.weight_decay,
                eps=cfg.eps
            )
        elif cfg.name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=cfg.betas[0]  # Use first beta as momentum
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.name}")
        
        return optimizer
    
    def _create_train_dataloader(self) -> DataLoader:
        """Create training DataLoader from config."""
        logger.info(f"Creating training dataloader from {self.config.data.train_data_path}")
        
        return get_batch_dataloader(
            data=self.config.data.train_data_path,
            batch_size=self.config.data.batch_size,
            context_length=self.config.data.context_length,
            device=self.device,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            shuffle=self.config.data.shuffle,
            vocab_size=self.config.model.vocab_size,
        )
    
    def _create_val_dataloader(self) -> DataLoader:
        """Create validation DataLoader from config."""
        if not self.config.data.valid_data_path:
            return None
        
        logger.info(f"Creating validation dataloader from {self.config.data.valid_data_path}")
        
        return get_batch_dataloader(
            data=self.config.data.valid_data_path,
            batch_size=self.config.data.batch_size,
            context_length=self.config.data.context_length,
            device=self.device,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            shuffle=False,  # Don't shuffle validation data
            vocab_size=self.config.model.vocab_size,
        )
    
    def _get_learning_rate(self, step: int) -> float:
        """Calculate learning rate for the given step."""
        cfg = self.config.scheduler
        
        if cfg.name.lower() == "cosine":
            return cosine_lr_schedule(
                current_t=step,
                lr_max=cfg.lr_max,
                lr_min=cfg.lr_min,
                total_warmup_t=cfg.warmup_steps,
                total_cos_anneal_t=cfg.total_steps
            )
        elif cfg.name.lower() == "constant":
            return cfg.lr_max
        elif cfg.name.lower() == "linear":
            if step < cfg.warmup_steps:
                return cfg.lr_max * step / cfg.warmup_steps
            else:
                return cfg.lr_max * max(0, 1 - (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps))
        elif cfg.name.lower() == "exponential":
            return cfg.lr_max * (cfg.lr_min / cfg.lr_max) ** (step / cfg.total_steps)
        else:
            logger.warning(f"Unknown scheduler: {cfg.name}, using constant lr")
            return cfg.lr_max
    
    def _update_learning_rate(self, step: int) -> float:
        """Update learning rate for all param groups."""
        lr = self._get_learning_rate(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def _set_optimizer_lr(self, lr: float) -> None:
        """Set learning rate for all optimizer parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _lr_finder_lr(self, step: int, min_lr: float, max_lr: float, total_steps: int, mode: str) -> float:
        """Return the LR for a range-test step."""
        if total_steps <= 1:
            return max_lr

        ratio = step / (total_steps - 1)
        if mode == "linear":
            return min_lr + ratio * (max_lr - min_lr)
        return min_lr * (max_lr / min_lr) ** ratio

    def _suggest_lr(
        self,
        lrs: list[float],
        losses: list[float],
        skip_begin: int,
        skip_end: int,
    ) -> float | None:
        """Pick the LR at the steepest loss descent, matching Lightning's usual heuristic."""
        if len(lrs) < 3 or len(lrs) != len(losses):
            return None

        start = min(max(skip_begin, 0), len(lrs) - 1)
        end = len(lrs) - min(max(skip_end, 0), len(lrs) - start - 1)
        if end - start < 2:
            start, end = 0, len(lrs)

        best_idx = None
        best_slope = 0.0
        for idx in range(start + 1, end):
            prev_lr = max(lrs[idx - 1], 1e-30)
            curr_lr = max(lrs[idx], 1e-30)
            denom = math.log10(curr_lr) - math.log10(prev_lr)
            if denom == 0:
                continue
            slope = (losses[idx] - losses[idx - 1]) / denom
            if best_idx is None or slope < best_slope:
                best_idx = idx
                best_slope = slope

        return lrs[best_idx] if best_idx is not None else None

    def lr_find(
        self,
        min_lr: float = 1e-8,
        max_lr: float = 1.0,
        num_training_steps: int = 100,
        mode: str = "exponential",
        early_stop_threshold: float | None = 4.0,
        update_attr: bool = True,
        skip_begin: int = 10,
        skip_end: int = 5,
    ) -> dict[str, float | int | list[float] | None]:
        """
        Run a short LR range test and restore model/optimizer state afterward.

        This mirrors PyTorch Lightning's LR finder workflow: sweep from min_lr to
        max_lr, record the smoothed loss curve, suggest the LR with the steepest
        descent, then optionally apply that LR to the trainer config.
        """
        if min_lr <= 0 or max_lr <= 0:
            raise ValueError("LR finder min_lr and max_lr must be positive")
        if max_lr <= min_lr:
            raise ValueError("LR finder max_lr must be greater than min_lr")
        if num_training_steps <= 0:
            raise ValueError("LR finder num_training_steps must be positive")

        logger.info(
            "Running LR finder from %.2e to %.2e for %d optimizer steps",
            min_lr,
            max_lr,
            num_training_steps,
        )

        model_state = copy.deepcopy(self.model.state_dict())
        optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        was_training = self.model.training
        original_step = self.global_step
        original_epoch = self.current_epoch

        lrs: list[float] = []
        losses: list[float] = []
        smoothed_loss = 0.0
        best_loss = float("inf")
        beta = 0.98
        finder_step = 0
        micro_step = 0

        self.model.train()
        self.optimizer.zero_grad()

        try:
            while finder_step < num_training_steps:
                made_progress = False
                for batch in tqdm(
                    self.train_dataloader,
                    total=num_training_steps,
                    initial=finder_step,
                    desc="LR finder",
                    leave=False,
                ):
                    if finder_step >= num_training_steps:
                        break

                    lr = self._lr_finder_lr(finder_step, min_lr, max_lr, num_training_steps, mode)
                    self._set_optimizer_lr(lr)

                    loss = self._training_step(batch)
                    micro_step += 1

                    if micro_step % self.config.training.gradient_accumulation_steps != 0:
                        continue

                    self._optimizer_step(finder_step)

                    loss_value = loss.item()
                    smoothed_loss = beta * smoothed_loss + (1 - beta) * loss_value
                    bias_corrected_loss = smoothed_loss / (1 - beta ** (finder_step + 1))

                    lrs.append(lr)
                    losses.append(bias_corrected_loss)
                    made_progress = True

                    if bias_corrected_loss < best_loss:
                        best_loss = bias_corrected_loss
                    elif (
                        early_stop_threshold is not None
                        and finder_step > skip_begin
                        and bias_corrected_loss > early_stop_threshold * best_loss
                    ):
                        logger.info(
                            "Stopping LR finder early at step %d because loss diverged",
                            finder_step + 1,
                        )
                        finder_step += 1
                        raise StopIteration

                    finder_step += 1

                if not made_progress:
                    raise RuntimeError("LR finder could not complete an optimizer step")

        except StopIteration:
            pass
        finally:
            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(optimizer_state)
            self.global_step = original_step
            self.current_epoch = original_epoch
            self.optimizer.zero_grad()
            self.model.train(was_training)

        suggestion = self._suggest_lr(lrs, losses, skip_begin=skip_begin, skip_end=skip_end)
        if suggestion is not None:
            logger.info("LR finder suggestion: %.6g", suggestion)
            if update_attr:
                self.config.optimizer.lr = suggestion
                self.config.scheduler.lr_max = suggestion
                self._set_optimizer_lr(suggestion)
                logger.info("Updated optimizer.lr and scheduler.lr_max to %.6g", suggestion)
        else:
            logger.warning("LR finder could not compute a learning-rate suggestion")

        result = {
            "suggested_lr": suggestion,
            "num_steps": len(lrs),
            "min_loss": min(losses) if losses else None,
            "lrs": lrs,
            "losses": losses,
        }

        if self.tracker.enabled:
            for idx, (lr, loss) in enumerate(zip(lrs, losses)):
                self.tracker.log_metrics(
                    {"lr_finder/lr": lr, "lr_finder/loss": loss},
                    idx,
                )

        return result
    
    def _calculate_total_steps(self) -> int:
        """Calculate total training steps."""
        if self.config.training.max_steps:
            return self.config.training.max_steps
        
        steps_per_epoch = len(self.train_dataloader)
        total_steps = steps_per_epoch * self.config.training.max_epochs
        return total_steps
    
    def _move_batch_to_device(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Move a dataloader batch to the training device in the main process."""
        inputs, targets = batch
        return inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
    
    def _training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Single training step."""
        inputs, targets = self._move_batch_to_device(batch)
        
        # Create token positions
        token_positions = torch.arange(inputs.shape[1], device=self.device)
        token_positions_expanded = repeat(token_positions, "seq -> batch seq", batch=inputs.shape[0])
        
        # Forward pass
        logits = self.model(inputs, token_positions_expanded)
        loss = cross_entropy_loss(logits, targets)
        
        # Backward pass (scaled by accumulation steps)
        loss = loss / self.config.training.gradient_accumulation_steps
        loss.backward()
        
        return loss * self.config.training.gradient_accumulation_steps
    
    def _optimizer_step(self, step: int) -> None:
        """Perform optimizer step with gradient clipping."""
        # Gradient clipping
        if self.config.training.gradient_clip_val:
            gradient_clipping(
                self.model.parameters(),
                self.config.training.gradient_clip_val
            )
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    @torch.no_grad()
    def _validation_step(self) -> float:
        """Run validation and return average loss."""
        if not self.val_dataloader:
            return None
        
        self.model.eval()
        val_losses = []
        max_val_batches = self.config.training.max_val_batches
        val_iter = self.val_dataloader
        if max_val_batches is not None:
            val_iter = islice(val_iter, max_val_batches)

        val_total = None
        if max_val_batches is not None:
            val_total = min(max_val_batches, len(self.val_dataloader))
        elif hasattr(self.val_dataloader, "__len__"):
            val_total = len(self.val_dataloader)
        
        for batch in tqdm(val_iter, total=val_total, desc="Validation", leave=False):
            inputs, targets = self._move_batch_to_device(batch)
            
            # Create token positions
            token_positions = torch.arange(inputs.shape[1], device=self.device)
            token_positions_expanded = repeat(token_positions, "seq -> batch seq", batch=inputs.shape[0])
            
            logits = self.model(inputs, token_positions_expanded)
            loss = cross_entropy_loss(logits, targets)
            val_losses.append(loss.item())
        
        self.model.train()
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0.0
        return avg_val_loss
    
    def save_checkpoint(self, suffix: str = "") -> Path:
        """Save a checkpoint."""
        checkpoint_name = f"checkpoint-epoch{self.current_epoch}-step{self.global_step}{suffix}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            iteration=self.global_step,
            out=checkpoint_path
        )
        
        # Clean up old checkpoints
        self._cleanup_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_checkpoints(self) -> None:
        """Keep only the last N checkpoints."""
        if self.config.checkpoint.keep_last_n <= 0:
            return
        
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint-*.pt"))
        
        if len(checkpoints) > self.config.checkpoint.keep_last_n:
            for checkpoint in checkpoints[:-self.config.checkpoint.keep_last_n]:
                logger.info(f"Removing old checkpoint: {checkpoint}")
                checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load a checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        iteration = load_checkpoint(
            src=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer
        )
        self.global_step = iteration
    
    def train(self, resume_from_checkpoint: str | Path | None = None) -> None:
        """
        Run the training loop.
        
        Args:
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        # Load checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        elif self.config.checkpoint.resume_from_checkpoint:
            self.load_checkpoint(self.config.checkpoint.resume_from_checkpoint)
        
        # Log configuration
        self.tracker.log_config(self.config.to_dict())

        if self.config.lr_finder.enabled and self.global_step == 0:
            self.lr_find(
                min_lr=self.config.lr_finder.min_lr,
                max_lr=self.config.lr_finder.max_lr,
                num_training_steps=self.config.lr_finder.num_training_steps,
                mode=self.config.lr_finder.mode,
                early_stop_threshold=self.config.lr_finder.early_stop_threshold,
                update_attr=self.config.lr_finder.update_attr,
                skip_begin=self.config.lr_finder.skip_begin,
                skip_end=self.config.lr_finder.skip_end,
            )
        
        # Calculate total steps
        total_steps = self._calculate_total_steps()
        self.config.scheduler.total_steps = total_steps
        logger.info(f"Starting training for {total_steps} steps")
        
        self.model.train()
        torch.manual_seed(self.config.training.seed)
        
        start_time = time.time()
        accumulated_loss = 0.0
        reached_time_limit = False
        
        for epoch in range(self.current_epoch, self.config.training.max_epochs):
            self.current_epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{self.config.training.max_epochs}")
            
            # Progress bar for training
            pbar = tqdm(self.train_dataloader, desc=f"Training Epoch {epoch + 1}")
            
            for batch_idx, batch in enumerate(pbar):
                # Check if we've reached max steps
                if self.config.training.max_steps and self.global_step >= self.config.training.max_steps:
                    logger.info(f"Reached max_steps ({self.config.training.max_steps})")
                    break
                
                # Training step
                loss = self._training_step(batch)
                accumulated_loss += loss.item()
                
                # Optimizer step
                if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                    self._optimizer_step(self.global_step)
                    
                    # Update learning rate
                    current_lr = self._update_learning_rate(self.global_step)
                    
                    self.global_step += 1
                    
                    # Logging
                    if self.global_step % self.config.training.log_interval == 0:
                        avg_loss = accumulated_loss / self.config.training.log_interval
                        elapsed = time.time() - start_time
                        throughput = self.global_step / elapsed
                        
                        log_msg = (
                            f"Step {self.global_step}/{total_steps} | "
                            f"Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.6f} | "
                            f"Throughput: {throughput:.2f} steps/sec"
                        )
                        logger.info(log_msg)
                        pbar.set_postfix({"loss": avg_loss, "lr": current_lr})
                        
                        self.train_losses.append(avg_loss)
                        
                        # Log metrics to experiment tracker
                        metrics = {
                            "train/loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/throughput_steps_per_sec": throughput,
                            "epoch": epoch,
                        }
                        self.tracker.log_metrics(metrics, self.global_step, elapsed)
                        
                        accumulated_loss = 0.0
                    
                    # Log gradient statistics
                    if self.config.experiment_tracking.log_gradients and self.global_step % self.config.training.log_interval == 0:
                        self.tracker.log_gradient_stats(self.model, self.global_step)
                    
                    # Validation
                    val_check_interval = self.config.training.val_check_interval
                    if val_check_interval and val_check_interval > 0 and self.global_step % val_check_interval == 0:
                        val_loss = self._validation_step()
                        if val_loss is not None:
                            logger.info(f"Validation Loss: {val_loss:.4f}")
                            self.val_losses.append(val_loss)
                            
                            # Log validation metrics
                            self.tracker.log_metrics(
                                {"val/loss": val_loss},
                                self.global_step,
                                time.time() - start_time
                            )
                            
                            # Save best checkpoint
                            if self.config.checkpoint.save_best_only and val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                ckpt_path = self.save_checkpoint(suffix="-best")
                                self.tracker.log_model_checkpoint(self.model, ckpt_path, self.global_step, is_best=True)
                    
                    # Regular checkpointing
                    if self.global_step % self.config.training.save_interval == 0:
                        ckpt_path = self.save_checkpoint()
                        self.tracker.log_model_checkpoint(self.model, ckpt_path, self.global_step, is_best=False)

                    if self.config.training.max_duration_seconds is not None:
                        elapsed = time.time() - start_time
                        if elapsed >= self.config.training.max_duration_seconds:
                            logger.info(
                                "Reached max_duration_seconds (%.1f) at step %d",
                                self.config.training.max_duration_seconds,
                                self.global_step,
                            )
                            reached_time_limit = True
                            break
            
            # Early exit if max_steps reached
            if reached_time_limit or (self.config.training.max_steps and self.global_step >= self.config.training.max_steps):
                break

        final_val_loss = self._validation_step()
        if final_val_loss is not None:
            logger.info(f"Final Validation Loss: {final_val_loss:.4f}")
            self.val_losses.append(final_val_loss)
            self.tracker.log_metrics(
                {"val/final_loss": final_val_loss},
                self.global_step,
                time.time() - start_time,
            )
        
        # Final checkpoint
        ckpt_path = self.save_checkpoint(suffix="-final")
        self.tracker.log_model_checkpoint(self.model, ckpt_path, self.global_step, is_best=False)
        
        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.2f}s ({elapsed/3600:.2f}h)")
        logger.info(f"Total steps: {self.global_step}")
        logger.info(f"Average throughput: {self.global_step / elapsed:.2f} steps/sec")
        
        # Finish experiment tracking
        self.tracker.finish()
    
    def get_training_summary(self) -> dict[str, any]:
        """Get a summary of training results."""
        return {
            "total_steps": self.global_step,
            "total_epochs": self.current_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss if self.val_losses else None,
            "model_params": sum(p.numel() for p in self.model.parameters()),
            "device": str(self.device),
            "dtype": str(self.torch_dtype),
        }
    
    def __repr__(self) -> str:
        return (
            f"Trainer(\n"
            f"  model={self.model.__class__.__name__},\n"
            f"  device={self.device},\n"
            f"  dtype={self.torch_dtype},\n"
            f"  global_step={self.global_step}\n"
            f")"
        )
