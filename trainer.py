"""Training module for SSRS semantic segmentation."""

import time
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from IPython.display import clear_output
from typing import Optional, Dict

from config import config
from model_wrapper import ModelWrapper
from utils import SemanticSegmentationDataset, CrossEntropy2d, accuracy, WEIGHTS
from evaluator import Evaluator


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
    
    def _setup_data_loader(self) -> None:
        """Setup training data loader."""
        print("Setting up training data...")
        print(f"Training IDs: {self.config.dataset.train_ids}")
        
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
            
            # Print progress every 100 iterations
            if batch_idx % 100 == 0:
                clear_output()
                self._print_batch_progress(
                    epoch, batch_idx, len(self.train_loader), 
                    loss.data.item(), data, output, target
                )
            
            # Clean up memory
            del data, target, loss
        
        training_time = time.time() - start_time
        avg_loss = epoch_loss / num_batches
        
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
        
        print(f'Train (epoch {epoch}/{self.config.dataset.epochs}) '
              f'[{batch_idx}/{total_batches} '
              f'({100. * batch_idx / total_batches:.0f}%)]\\t'
              f'Loss: {loss_value:.6f}\\t'
              f'Accuracy: {batch_accuracy:.2f}%')
    
    def validate(self, epoch: int) -> float:
        """Validate the model.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Validation mIoU score
        """
        if self.evaluator is None:
            print("No evaluator provided, skipping validation")
            return 0.0
        
        print("Running validation...")
        start_time = time.time()
        
        self.model.eval()
        miou = self.evaluator.evaluate(
            self.model_wrapper,
            test_ids=self.config.dataset.test_ids,
            stride=self.config.dataset.stride_size
        )
        self.model.train()
        
        validation_time = time.time() - start_time
        print(f"Validation time: {validation_time:.3f} seconds")
        print(f"Validation mIoU: {miou:.4f}")
        
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
            print(f"New best model saved: {checkpoint_path}")
    
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
            
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Total epochs: {epochs}")
        print(f"Validation frequency: every {save_epoch} epochs")
        print(f"Initial best mIoU threshold: {self.best_miou}")
        
        for epoch in range(1, epochs + 1):
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            print(f"\\nEpoch {epoch}/{epochs} completed")
            print(f"Training time: {train_metrics['training_time']:.3f} seconds")
            print(f"Average loss: {train_metrics['avg_loss']:.6f}")
            
            # Validate and save model
            if epoch % save_epoch == 0:
                miou = self.validate(epoch)
                self.save_checkpoint(epoch, miou)
        
        print("=" * 60)
        print("TRAINING COMPLETED")
        print(f"Best mIoU achieved: {self.best_miou:.4f}")
        print("=" * 60)
