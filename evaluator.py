"""Evaluation module for SSRS semantic segmentation."""

import os
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from typing import List, Tuple, Optional, Union
from skimage import io
from IPython.display import clear_output

from config import config
from utils import (
    sliding_window, count_sliding_window, grouper, 
    convert_from_color, convert_to_color, metrics
)


class Evaluator:
    """Evaluator class for semantic segmentation model."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.config = config
    
    def _load_test_data(self, test_ids: List[str]) -> Tuple[List, List, List, List]:
        """Load test data including images, DSM, labels, and eroded labels.
        
        Args:
            test_ids: List of test image IDs
            
        Returns:
            Tuple of (test_images, test_dsms, test_labels, eroded_labels)
        """
        test_images = []
        test_dsms = []
        test_labels = []
        eroded_labels = []
        
        for id_ in test_ids:
            # Load image data
            try:
                data_path = os.path.join(self.config.dataset.data_root, 
                                       self.config.dataset.data_pattern.format(id_))
                img_data = io.imread(data_path)
                
                # Handle different image formats
                if len(img_data.shape) == 3 and img_data.shape[2] >= 3:
                    # Use first 3 channels (RGB)
                    img_data = img_data[:, :, :3]
                elif len(img_data.shape) == 2:
                    # Grayscale - repeat to 3 channels
                    img_data = np.repeat(img_data[:, :, None], 3, axis=2)
                
                test_images.append(1 / 255 * np.asarray(img_data, dtype='float32'))
            except Exception as e:
                print(f"Error loading image {id_}: {e}")
                test_images.append(np.zeros((256, 256, 3), dtype='float32'))
            
            # Load DSM data
            try:
                dsm_path = os.path.join(self.config.dataset.data_root,
                                      self.config.dataset.dsm_pattern.format(id_))
                dsm_data = np.asarray(io.imread(dsm_path), dtype='float32')
                if len(dsm_data.shape) > 2:
                    dsm_data = dsm_data[:, :, 0]  # Take first channel
                test_dsms.append(dsm_data)
            except Exception as e:
                print(f"Error loading DSM {id_}: {e}")
                test_dsms.append(np.zeros((256, 256), dtype='float32'))
            
            # Load label data
            try:
                label_path = os.path.join(self.config.dataset.data_root,
                                        self.config.dataset.label_pattern.format(id_))
                label_data = np.asarray(io.imread(label_path), dtype='uint8')
                test_labels.append(label_data)
            except Exception as e:
                print(f"Error loading label {id_}: {e}")
                test_labels.append(np.zeros((256, 256), dtype='uint8'))
            
            # Load eroded labels (if available)
            try:
                if self.config.dataset.eroded_pattern:
                    eroded_path = os.path.join(self.config.dataset.data_root,
                                             self.config.dataset.eroded_pattern.format(id_))
                    eroded_data = convert_from_color(io.imread(eroded_path))
                else:
                    # Use regular labels if eroded not available
                    eroded_data = convert_from_color(test_labels[-1]) if len(test_labels[-1].shape) == 3 else test_labels[-1]
                eroded_labels.append(eroded_data)
            except Exception as e:
                print(f"Error loading eroded label {id_}: {e}")
                # Fallback to regular labels
                eroded_data = convert_from_color(test_labels[-1]) if len(test_labels[-1].shape) == 3 else test_labels[-1]
                eroded_labels.append(eroded_data)
        
        return test_images, test_dsms, test_labels, eroded_labels
    
    def _normalize_dsm(self, dsm: np.ndarray) -> np.ndarray:
        """Normalize DSM data to [0, 1] range.
        
        Args:
            dsm: Input DSM array
            
        Returns:
            Normalized DSM array
        """
        dsm_min = np.min(dsm)
        dsm_max = np.max(dsm)
        return (dsm - dsm_min) / (dsm_max - dsm_min)
    
    def _predict_image(self, model, img: np.ndarray, dsm: np.ndarray, 
                      stride: int, batch_size: int, 
                      window_size: Tuple[int, int]) -> np.ndarray:
        """Predict segmentation for a single image using sliding window.
        
        Args:
            model: The model to use for prediction
            img: Input image
            dsm: Input DSM
            stride: Stride for sliding window
            batch_size: Batch size for inference
            window_size: Window size for patches
            
        Returns:
            Predicted segmentation map
        """
        pred = np.zeros(img.shape[:2] + (self.config.n_classes,))
        
        # Normalize DSM
        dsm_normalized = self._normalize_dsm(dsm)
        
        # Calculate total number of windows
        total_windows = count_sliding_window(img, step=stride, window_size=window_size)
        total_batches = total_windows // batch_size
        
        # Process image in batches using sliding window
        for i, coords in enumerate(
            tqdm(
                grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)),
                total=total_batches,
                leave=False,
                desc="Processing patches"
            )
        ):
            # Prepare image patches
            image_patches = [
                np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) 
                for x, y, w, h in coords
            ]
            image_patches = np.asarray(image_patches)
            image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
            
            # Prepare DSM patches
            dsm_patches = [
                np.copy(dsm_normalized[x:x + w, y:y + h]) 
                for x, y, w, h in coords
            ]
            dsm_patches = np.asarray(dsm_patches)
            dsm_patches = Variable(torch.from_numpy(dsm_patches).cuda(), volatile=True)
            
            # Forward pass
            with torch.no_grad():
                outputs = model.forward(image_patches, dsm_patches, mode='Test')
                outputs = outputs.data.cpu().numpy()
            
            # Accumulate predictions
            for output, (x, y, w, h) in zip(outputs, coords):
                output = output.transpose((1, 2, 0))
                pred[x:x + w, y:y + h] += output
            
            del outputs
        
        # Convert to class predictions
        return np.argmax(pred, axis=-1)
    
    def evaluate(self, model_wrapper, test_ids: List[str], 
                stride: int, batch_size: Optional[int] = None,
                window_size: Optional[Tuple[int, int]] = None,
                return_predictions: bool = False) -> Union[float, Tuple[float, List, List]]:
        """Evaluate the model on test data.
        
        Args:
            model_wrapper: Wrapped model for evaluation
            test_ids: List of test image IDs
            stride: Stride for sliding window inference
            batch_size: Batch size for inference (uses config if None)
            window_size: Window size for patches (uses config if None)
            return_predictions: Whether to return prediction arrays
            
        Returns:
            mIoU score, or tuple of (mIoU, predictions, ground_truths) if return_predictions=True
        """
        if batch_size is None:
            batch_size = self.config.training.batch_size
        if window_size is None:
            window_size = self.config.training.window_size
        
        print(f"Evaluating on {len(test_ids)} test images...")
        print(f"Test IDs: {test_ids}")
        
        # Load test data
        test_images, test_dsms, test_labels, eroded_labels = self._load_test_data(test_ids)
        
        all_predictions = []
        all_ground_truths = []
        
        model = model_wrapper.get_model()
        model.eval()
        
        # Process each test image
        with torch.no_grad():
            for idx, (img, dsm, gt, gt_eroded) in enumerate(
                tqdm(
                    zip(test_images, test_dsms, test_labels, eroded_labels),
                    total=len(test_ids),
                    desc="Evaluating images"
                )
            ):
                print(f"\\nProcessing image {test_ids[idx]}...")
                
                # Predict segmentation
                pred = self._predict_image(
                    model_wrapper, img, dsm, stride, batch_size, window_size
                )
                
                all_predictions.append(pred)
                all_ground_truths.append(gt_eroded)
                
                clear_output(wait=True)
        
        # Calculate metrics
        print("Calculating metrics...")
        miou = metrics(
            np.concatenate([p.ravel() for p in all_predictions]),
            np.concatenate([gt.ravel() for gt in all_ground_truths])
        )
        
        if return_predictions:
            return miou, all_predictions, all_ground_truths
        else:
            return miou
    
    def save_predictions(self, predictions: List[np.ndarray], test_ids: List[str],
                        output_prefix: str = "inference") -> None:
        """Save prediction results as colored images.
        
        Args:
            predictions: List of prediction arrays
            test_ids: List of test image IDs
            output_prefix: Prefix for output filenames
        """
        output_dir = self.config.get_output_dir()
        
        for pred, test_id in zip(predictions, test_ids):
            # Convert predictions to color
            colored_pred = convert_to_color(pred)
            
            # Save image
            output_path = f"{output_dir}{output_prefix}_{self.config.model_name}_{test_id}.png"
            io.imsave(output_path, colored_pred)
            print(f"Saved prediction: {output_path}")
    
    def evaluate_and_save(self, model_wrapper, test_ids: List[str],
                         stride: int = 32, save_predictions: bool = True) -> float:
        """Evaluate model and optionally save predictions.
        
        Args:
            model_wrapper: Wrapped model for evaluation
            test_ids: List of test image IDs
            stride: Stride for sliding window inference
            save_predictions: Whether to save prediction images
            
        Returns:
            mIoU score
        """
        # Evaluate with predictions
        miou, predictions, ground_truths = self.evaluate(
            model_wrapper, test_ids, stride, return_predictions=True
        )
        
        print(f"\\nEvaluation Results:")
        print(f"mIoU: {miou:.4f}")
        
        # Save predictions if requested
        if save_predictions:
            print("\\nSaving predictions...")
            self.save_predictions(predictions, test_ids)
        
        return miou
