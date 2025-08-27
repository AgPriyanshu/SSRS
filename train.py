"""
SSRS Semantic Segmentation Training Script

This is the main training script that has been refactored for better organization.
For the clean, modular version, please use train_clean.py

This script maintains compatibility with the original codebase.
"""

import torch
import torch.optim as optim
from utils2 import *
from config import DatasetConfig
from config import config
from model_wrapper import ModelWrapper
from trainer import Trainer
from evaluator import Evaluator

# Setup default dataset configuration for backward compatibility
if config.dataset is None or config.dataset.name == "default":
    # Create a legacy-compatible dataset configuration
    legacy_dataset = DatasetConfig(
        name=DATASET,
        train_ids=train_ids,
        test_ids=test_ids,
        stride_size=Stride_Size,
        epochs=epochs,
        save_epoch=save_epoch,
        data_root="./data/",
        data_pattern="images/img_{}.jpg",
        dsm_pattern="dsm/dsm_{}.tif",
        label_pattern="labels/label_{}.png"
    )
    config.set_dataset_config(legacy_dataset)

# Initialize configuration
config.print_config()

# Initialize model
print("Initializing model...")
model_wrapper = ModelWrapper(num_classes=N_CLASSES)
model_wrapper.print_parameter_summary()

# Get the underlying model for compatibility
net = model_wrapper.get_model()

print("Training IDs:", train_ids)
print("Testing IDs:", test_ids)

# Setup data loader - use new dataset class with backward compatibility
train_set = SemanticSegmentationDataset(
    ids=train_ids,
    data_pattern=config.dataset.data_pattern,
    dsm_pattern=config.dataset.dsm_pattern,
    label_pattern=config.dataset.label_pattern,
    data_root=config.dataset.data_root,
    cache=CACHE
)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)

# Setup optimizer with different learning rates for encoder/decoder
base_lr = 0.01
params_dict = dict(net.named_parameters())
params = []
for key, value in params_dict.items():
    if '_D' in key:
        # Decoder weights are trained at the nominal learning rate
        params += [{'params': [value], 'lr': base_lr}]
    else:
        # Encoder weights are trained at lr / 2 (we have VGG-16 weights as initialization)
        params += [{'params': [value], 'lr': base_lr / 2}]

optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [25, 35, 45], gamma=0.1)

# Legacy test function - using new Evaluator class for better organization
def test(net, test_ids, all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    """Legacy test function for backward compatibility."""
    # Create temporary model wrapper and evaluator
    temp_model_wrapper = ModelWrapper(num_classes=N_CLASSES)
    temp_model_wrapper.model = net  # Use the existing network
    
    evaluator = Evaluator()
    
    if all:
        miou, all_preds, all_gts = evaluator.evaluate(
            temp_model_wrapper, test_ids, stride, batch_size, window_size, return_predictions=True
        )
        return miou, all_preds, all_gts
    else:
        miou = evaluator.evaluate(temp_model_wrapper, test_ids, stride, batch_size, window_size)
        return miou


# Legacy train function - using new Trainer class for better organization
def train(net, optimizer, epochs, scheduler=None, weights=WEIGHTS, save_epoch=1):
    """Legacy train function for backward compatibility."""
    # Create temporary model wrapper, evaluator, and trainer
    temp_model_wrapper = ModelWrapper(num_classes=N_CLASSES)
    temp_model_wrapper.model = net  # Use the existing network
    
    evaluator = Evaluator()
    trainer = Trainer(temp_model_wrapper, evaluator)
    
    # Use the existing optimizer and scheduler
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.train_loader = train_loader
    
    # Run training
    trainer.train(epochs=epochs, save_epoch=save_epoch)

# Main execution logic
def main():
    """Main execution function."""
    if MODE == 'Train':
        print("Starting training...")
        train(net, optimizer, epochs, scheduler, weights=WEIGHTS, save_epoch=save_epoch)
        
    elif MODE == 'Test':
        print("Starting testing...")
        
        # Load pre-trained model
        output_dir = config.get_output_dir()
        model_path = f"{output_dir}YOUR_MODEL"
        
        try:
            print(f"Loading model from: {model_path}")
            net.load_state_dict(torch.load(model_path), strict=False)
            net.eval()
            
            # Run evaluation
            miou, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
            print(f"Test mIoU: {miou:.4f}")
            
            # Save predictions
            for pred, test_id in zip(all_preds, test_ids):
                colored_pred = convert_to_color(pred)
                output_path = f"{output_dir}inference_UNetFormer_{config.model_name}_tile_{test_id}.png"
                io.imsave(output_path, colored_pred)
                print(f"Saved prediction: {output_path}")
                
        except FileNotFoundError:
            print(f"Model file not found: {model_path}")
            print("Please specify a valid model path for testing.")
        except Exception as e:
            print(f"Error during testing: {e}")
    
    else:
        print(f"Unknown mode: {MODE}. Please use 'Train' or 'Test'.")


if __name__ == "__main__":
    main()

