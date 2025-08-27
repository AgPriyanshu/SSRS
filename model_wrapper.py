"""Model wrapper for UNetFormer with parameter analysis."""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from unetformer_mmsam import UNetFormer
from config import config


class ModelWrapper:
    """Wrapper class for UNetFormer model with utilities."""
    
    def __init__(self, num_classes: int = None):
        """Initialize the model wrapper.
        
        Args:
            num_classes: Number of output classes. If None, uses config value.
        """
        if num_classes is None:
            num_classes = config.n_classes
            
        self.num_classes = num_classes
        self.model = None
        self._create_model()
        
    def _create_model(self) -> None:
        """Create and initialize the model."""
        self.model = UNetFormer(num_classes=self.num_classes).cuda()
        
    def get_parameter_counts(self) -> Dict[str, int]:
        """Get detailed parameter counts for different parts of the model.
        
        Returns:
            Dictionary containing parameter counts for different model components.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        total_params = 0
        encoder_params = 0
        lora_params = 0
        
        # Count total parameters
        for name, param in self.model.named_parameters():
            total_params += param.nelement()
            
        # Count encoder and LoRA parameters
        for name, param in self.model.image_encoder.named_parameters():
            if "lora_" not in name:
                encoder_params += param.nelement()
            else:
                lora_params += param.nelement()
                
        other_params = total_params - encoder_params - lora_params
        
        return {
            "total": total_params,
            "image_encoder": encoder_params,
            "lora": lora_params,
            "others": other_params
        }
    
    def print_parameter_summary(self) -> None:
        """Print a summary of model parameters."""
        param_counts = self.get_parameter_counts()
        
        print("=" * 50)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 50)
        print(f"Total Parameters:     {param_counts['total']:,}")
        print(f"Image Encoder:        {param_counts['image_encoder']:,}")
        print(f"LoRA Parameters:      {param_counts['lora']:,}")
        print(f"Other Parameters:     {param_counts['others']:,}")
        print("=" * 50)
    
    def get_model(self) -> nn.Module:
        """Get the underlying model."""
        return self.model
    
    def load_weights(self, checkpoint_path: str, strict: bool = False) -> None:
        """Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            strict: Whether to strictly enforce that the keys match.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        try:
            self.model.load_state_dict(torch.load(checkpoint_path), strict=strict)
            print(f"Successfully loaded weights from: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise
    
    def save_weights(self, save_path: str) -> None:
        """Save model weights to file.
        
        Args:
            save_path: Path where to save the model weights.
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        try:
            torch.save(self.model.state_dict(), save_path)
            print(f"Model weights saved to: {save_path}")
        except Exception as e:
            print(f"Error saving weights: {e}")
            raise
    
    def set_train_mode(self) -> None:
        """Set model to training mode."""
        if self.model is not None:
            self.model.train()
    
    def set_eval_mode(self) -> None:
        """Set model to evaluation mode."""
        if self.model is not None:
            self.model.eval()
    
    def forward(self, image_patches: torch.Tensor, dsm_patches: torch.Tensor, mode: str = 'Train') -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            image_patches: Input image patches
            dsm_patches: Input DSM patches
            mode: Forward mode ('Train' or 'Test')
            
        Returns:
            Model output tensor
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
            
        return self.model(image_patches, dsm_patches, mode=mode)
