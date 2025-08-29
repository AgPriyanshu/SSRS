"""Training module for SSRS semantic segmentation."""

import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from typing import Optional, Dict
import warnings
import sys
from datetime import datetime

from config import config
from dataset import SemanticSegmentationDataset
from multi_dataset import MultiDatasetSemanticSegmentation
from model_wrapper import ModelWrapper
from evaluator import Evaluator
from utils2 import WEIGHTS, CrossEntropy2d, accuracy

# Suppress specific PyTorch warnings for cleaner output
warnings.filterwarnings("ignore", message=".*size_average and reduce args will be deprecated.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered in scalar divide.*")


class Trainer:
    """Trainer class for semantic segmentation model."""
    
    def __init__(self, model_wrapper: ModelWrapper, evaluator: Optional[Evaluator] = None):
        """Initialize the trainer.
        
        Args:
            model_wrapper: Wrapped model for training
            evaluator: Evaluator for validation (optional)
        """
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.get_model()
        self.evaluator = evaluator
        
        # Training configuration
        self.config = config
        self.best_miou = 0.82
        
        # Initialize training components
        self._setup_data_loader()
        self._setup_optimizer()
        self._setup_loss_function()
        
        # Training metrics
        self.losses = np.zeros(1000000)
        self.mean_losses = np.zeros(100000000)
        
        # Early monitoring
        self.recent_losses = []
        self.recent_accuracies = []
        self.loss_trend_window = 20  # Track last 20 batches for trend
        self.early_stop_patience = 50  # Stop if no improvement in 50 batches
    
    def _setup_data_loader(self) -> None:
        """Setup training data loader."""
        print("Setting up training data...")
        print(f"Training IDs: {self.config.dataset.train_ids}")
        
        # Detect if using multi-dataset configuration
        is_multi_dataset = (
            self.config.dataset.name == "multi_dataset_combined" or
            any("_" in str(id) and not str(id).startswith("img_") for id in self.config.dataset.train_ids[:5])
        )
        
        if is_multi_dataset:
            print("🔗 Using MULTI-DATASET loader for maximum diversity!")
            train_dataset = MultiDatasetSemanticSegmentation(
                ids=self.config.dataset.train_ids,
                mapping_file="multi_dataset_mapping.json",
                data_root=self.config.dataset.data_root,
                cache=self.config.training.cache,
                augmentation=True
            )
        else:
            print("📁 Using single dataset loader")
            train_dataset = SemanticSegmentationDataset(
                ids=self.config.dataset.train_ids,
                data_pattern=self.config.dataset.data_pattern,
                dsm_pattern=self.config.dataset.dsm_pattern,
                label_pattern=self.config.dataset.label_pattern,
                data_root=self.config.dataset.data_root,
                cache=self.config.training.cache
            )
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=self.config.training.batch_size
        )
        
        print(f"Training dataset loaded: {len(train_dataset)} samples")
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Create parameter groups with different learning rates
        base_lr = self.config.training.base_lr
        params = []
        
        for name, param in self.model.named_parameters():
            if '_D' in name:
                # Decoder weights use nominal learning rate
                params.append({'params': [param], 'lr': base_lr})
            else:
                # Encoder weights use half learning rate
                params.append({'params': [param], 'lr': base_lr / 2})
        
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=base_lr,
            momentum=self.config.training.momentum,
            weight_decay=self.config.training.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=self.config.training.scheduler_milestones,
            gamma=self.config.training.scheduler_gamma
        )
        
        print(f"Optimizer configured with base LR: {base_lr}")
    
    def _setup_loss_function(self) -> None:
        """Setup loss function with class weights."""
        self.weights = WEIGHTS.cuda()
        print("Loss function configured with class weights")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        start_time = time.time()
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, dsm, target) in enumerate(self.train_loader):
            # Move data to GPU
            data = Variable(data.cuda())
            dsm = Variable(dsm.cuda())
            target = Variable(target.cuda())
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data, dsm, mode='Train')
            loss = CrossEntropy2d(output, target, weight=self.weights)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            iter_idx = (epoch - 1) * len(self.train_loader) + batch_idx
            self.losses[iter_idx] = loss.data.item()
            self.mean_losses[iter_idx] = np.mean(
                self.losses[max(0, iter_idx - 100):iter_idx]
            )
            
            epoch_loss += loss.data.item()
            num_batches += 1
            
            # Track recent performance for early detection
            current_loss = loss.data.item()
            pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
            gt = target.data.cpu().numpy()[0]
            current_acc = accuracy(pred, gt)
            
            self.recent_losses.append(current_loss)
            self.recent_accuracies.append(current_acc)
            
            # Keep only recent window
            if len(self.recent_losses) > self.loss_trend_window:
                self.recent_losses.pop(0)
                self.recent_accuracies.pop(0)
            
            # Print progress every 10 iterations for faster feedback
            if batch_idx % 10 == 0:
                self._print_batch_progress_with_trends(
                    epoch, batch_idx, len(self.train_loader), 
                    current_loss, current_acc, data, output, target
                )
                
                # Check for early overfitting signs
                if batch_idx > 30:  # Only after some batches
                    self._check_early_overfitting_signs(epoch, batch_idx)
            
            # Clean up memory
            del data, target, loss
        
        training_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        
        # Print newline after progress bar completes
        print()
        
        return {
            'avg_loss': avg_loss,
            'training_time': training_time
        }
    
    def _print_batch_progress(self, epoch: int, batch_idx: int, total_batches: int,
                             loss_value: float, data: torch.Tensor, 
                             output: torch.Tensor, target: torch.Tensor) -> None:
        """Print training progress for current batch."""
        # Calculate accuracy for current batch
        pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
        gt = target.data.cpu().numpy()[0]
        batch_accuracy = accuracy(pred, gt)
        
        # Create progress bar
        progress = batch_idx / total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Format output with better structure
        sys.stdout.write('\r')
        sys.stdout.write(
            f'Epoch {epoch:2d}/{self.config.dataset.epochs} '
            f'[{bar}] '
            f'{batch_idx:4d}/{total_batches} '
            f'({progress*100:5.1f}%) | '
            f'Loss: {loss_value:7.4f} | '
            f'Acc: {batch_accuracy:5.1f}% | '
            f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
        )
        sys.stdout.flush()
    
    def _print_batch_progress_with_trends(self, epoch: int, batch_idx: int, total_batches: int,
                                         loss_value: float, accuracy_value: float,
                                         data: torch.Tensor, output: torch.Tensor, target: torch.Tensor) -> None:
        """Print training progress with trend analysis."""
        # Create progress bar
        progress = batch_idx / total_batches
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        
        # Calculate trends
        avg_recent_loss = np.mean(self.recent_losses) if self.recent_losses else loss_value
        avg_recent_acc = np.mean(self.recent_accuracies) if self.recent_accuracies else accuracy_value
        
        # Trend indicators
        loss_trend = ""
        acc_trend = ""
        if len(self.recent_losses) >= 10:
            # Simple trend detection
            recent_half = self.recent_losses[-5:]
            earlier_half = self.recent_losses[-10:-5]
            if np.mean(recent_half) < np.mean(earlier_half):
                loss_trend = "↓"  # Improving
            elif np.mean(recent_half) > np.mean(earlier_half):
                loss_trend = "↑"  # Worsening
            else:
                loss_trend = "→"  # Stable
                
            recent_acc_half = self.recent_accuracies[-5:]
            earlier_acc_half = self.recent_accuracies[-10:-5]
            if np.mean(recent_acc_half) > np.mean(earlier_acc_half):
                acc_trend = "↑"  # Improving
            elif np.mean(recent_acc_half) < np.mean(earlier_acc_half):
                acc_trend = "↓"  # Worsening
            else:
                acc_trend = "→"  # Stable
        
        # Warning indicators (adjusted for aggressive settings)
        warning = ""
        if accuracy_value > 90:  # Lowered threshold
            warning = "⚠️ "
        elif accuracy_value > 85:  # Lowered threshold
            warning = "🔶"
        
        # Format output with trends
        sys.stdout.write('\r')
        sys.stdout.write(
            f'{warning}Epoch {epoch:2d}/{self.config.dataset.epochs} '
            f'[{bar}] '
            f'{batch_idx:4d}/{total_batches} '
            f'({progress*100:5.1f}%) | '
            f'Loss: {loss_value:6.4f}{loss_trend} (avg:{avg_recent_loss:6.4f}) | '
            f'Acc: {accuracy_value:5.1f}%{acc_trend} (avg:{avg_recent_acc:5.1f}%) | '
            f'LR: {self.optimizer.param_groups[0]["lr"]:.2e}'
        )
        sys.stdout.flush()
    
    def _check_early_overfitting_signs(self, epoch: int, batch_idx: int) -> None:
        """Check for early signs of overfitting and provide warnings."""
        if len(self.recent_accuracies) < 10:
            return
            
        avg_acc = np.mean(self.recent_accuracies)
        max_acc = np.max(self.recent_accuracies)
        
        # Strong overfitting signals with auto-stop
        if avg_acc > 99:
            print(f"\n🚨 CRITICAL: Average accuracy {avg_acc:.1f}% indicates severe overfitting!")
            print("   AUTOMATIC STOP: Training halted to prevent overfitting")
            raise KeyboardInterrupt("Auto-stopped due to severe overfitting")
            
        elif avg_acc > 95:
            print(f"\n⚠️  WARNING: High accuracy {avg_acc:.1f}% - possible overfitting")
            print("   Monitor validation closely")
            
        elif max_acc == 100 and avg_acc > 85:  # Lowered threshold
            print(f"\n🔶 CAUTION: Perfect accuracy detected with high average ({avg_acc:.1f}%)")
            
        # Additional check for new aggressive settings
        elif avg_acc > 85 and batch_idx < 100:  # Early high accuracy
            print(f"\n⚠️  EARLY HIGH ACCURACY: {avg_acc:.1f}% too high too fast")
            print("   This suggests the learning rate is still too high")
    
    def mini_validation(self, num_samples: int = 3) -> float:
        """Run quick validation on a small subset for early feedback."""
        if self.evaluator is None:
            return 0.0
            
        # Use only first few test samples for speed
        mini_test_ids = self.config.dataset.test_ids[:num_samples]
        
        print(f"\n🔍 Mini-validation on {num_samples} samples...")
        start_time = time.time()
        
        self.model.eval()
        miou = self.evaluator.evaluate(
            self.model_wrapper,
            test_ids=mini_test_ids,
            stride=self.config.dataset.stride_size
        )
        self.model.train()
        
        validation_time = time.time() - start_time
        print(f"   Mini-validation mIoU: {miou:.4f} (time: {validation_time:.1f}s)")
        
        return miou
    
    def validate(self, epoch: int) -> float:
        """Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Validation mIoU score
        """
        if self.evaluator is None:
            print("❌ No evaluator provided, skipping validation")
            return 0.0
        
        print(f"📊 Evaluating on {len(self.config.dataset.test_ids)} test images...")
        start_time = time.time()
        
        self.model.eval()
        miou = self.evaluator.evaluate(
            self.model_wrapper,
            test_ids=self.config.dataset.test_ids,
            stride=self.config.dataset.stride_size
        )
        self.model.train()
        
        validation_time = time.time() - start_time
        
        print("📈 Validation Results:")
        print(f"   Time:  {validation_time:6.2f}s")
        print(f"   mIoU:  {miou:6.4f}")
        if miou > self.best_miou:
            print(f"   🎉 New best mIoU! (previous: {self.best_miou:.4f})")
        
        return miou
    
    def save_checkpoint(self, epoch: int, miou: float) -> None:
        """Save model checkpoint if performance improved.
        
        Args:
            epoch: Current epoch number
            miou: Current mIoU score
        """
        if miou > self.best_miou:
            output_dir = self.config.get_output_dir()
            checkpoint_path = f"{output_dir}{self.config.model_name}_epoch{epoch}_{miou:.4f}"
            
            self.model_wrapper.save_weights(checkpoint_path)
            self.best_miou = miou
            print(f"💾 Model checkpoint saved: {checkpoint_path}")
        else:
            print(f"   No improvement (best: {self.best_miou:.4f})")
    
    def train(self, epochs: Optional[int] = None, save_epoch: Optional[int] = None) -> None:
        """Main training loop.
        
        Args:
            epochs: Number of epochs to train (uses config if None)
            save_epoch: Frequency of validation and saving (uses config if None)
        """
        if epochs is None:
            epochs = self.config.dataset.epochs
        if save_epoch is None:
            save_epoch = self.config.dataset.save_epoch
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 60)
        print("🚀 STARTING TRAINING")
        print("=" * 60)
        print(f"Started at:           {timestamp}")
        print(f"Total epochs:         {epochs}")
        print(f"Validation frequency: every {save_epoch} epochs")
        print(f"Best mIoU threshold:  {self.best_miou:.4f}")
        print(f"Training samples:     {len(self.train_loader.dataset)}")
        print(f"Batches per epoch:    {len(self.train_loader)}")
        print("=" * 60)
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Update learning rate after training
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch:2d}/{epochs} SUMMARY")
            print(f"{'='*60}")
            print(f"Training time:    {train_metrics['training_time']:8.2f}s")
            print(f"Epoch total time: {epoch_time:8.2f}s")
            print(f"Average loss:     {train_metrics['avg_loss']:8.4f}")
            print(f"Learning rate:    {current_lr:.2e}")
            
            # Early performance check (quick validation on subset)
            if epoch == 1:  # First epoch - get early feedback
                mini_miou = self.mini_validation(num_samples=2)
                if mini_miou > 0:
                    print(f"   Early indicator: mini-mIoU = {mini_miou:.4f}")
            
            # Full validation and save model
            if epoch % save_epoch == 0:
                print("\n🔍 Running full validation...")
                miou = self.validate(epoch)
                self.save_checkpoint(epoch, miou)
            
            print(f"{'='*60}")
        
        end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED")
        print("=" * 60)
        print(f"Completed at:         {end_timestamp}")
        print(f"Best mIoU achieved:   {self.best_miou:.4f}")
        print(f"Total epochs:         {epochs}")
        print("=" * 60)
