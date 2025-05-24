"""
SAM2 Segmenter Base Class

This module provides the base SAM2 segmentation functionality for video analysis.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from pathlib import Path

try:
    from sam2.build_sam import build_sam2_video_predictor, build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
except ImportError:
    print("Warning: SAM2 not installed. Please install segment-anything-2")
    # Create dummy classes for development
    class SAM2ImagePredictor:
        pass
    class SAM2VideoPredictor:
        pass


class SAM2Segmenter:
    """
    Base SAM2 segmentation class for video analysis.
    
    This class provides core SAM2 functionality including model loading,
    frame processing, and basic segmentation capabilities.
    """
    
    def __init__(
        self,
        model_checkpoint: str = "sam2_hiera_large.pt",
        model_config: str = "sam2_hiera_l.yaml",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.0
    ):
        """
        Initialize the SAM2 segmenter.
        
        Args:
            model_checkpoint: Path to SAM2 model checkpoint
            model_config: Path to SAM2 model configuration
            device: Device to run inference on
            confidence_threshold: Minimum confidence for segmentations
            mask_threshold: Threshold for mask binarization
        """
        self.model_checkpoint = model_checkpoint
        self.model_config = model_config
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize models
        self.image_predictor = None
        self.video_predictor = None
        self.inference_state = None
        
        # Video processing state
        self.current_video_path = None
        self.frame_idx = 0
        
        # Initialize models
        self._load_models()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_models(self):
        """Load SAM2 models for image and video processing."""
        try:
            # Load image predictor
            sam2_model = build_sam2(self.model_config, self.model_checkpoint, device=self.device)
            self.image_predictor = SAM2ImagePredictor(sam2_model)
            
            # Load video predictor
            self.video_predictor = build_sam2_video_predictor(
                self.model_config, 
                self.model_checkpoint,
                device=self.device
            )
            
            self.logger.info("SAM2 models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load SAM2 models: {e}")
            # Create dummy predictors for development
            self.image_predictor = SAM2ImagePredictor()
            self.video_predictor = SAM2VideoPredictor()
    
    def set_video(self, video_path: str) -> bool:
        """
        Set up video for processing.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if hasattr(self.video_predictor, 'init_state'):
                self.inference_state = self.video_predictor.init_state(video_path=video_path)
                self.current_video_path = video_path
                self.frame_idx = 0
                self.logger.info(f"Video initialized: {video_path}")
                return True
            else:
                # Fallback for development
                self.current_video_path = video_path
                self.frame_idx = 0
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to initialize video: {e}")
            return False
    
    def segment_frame(
        self,
        frame: np.ndarray,
        prompts: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Segment a single frame using SAM2.
        
        Args:
            frame: Input frame (BGR format)
            prompts: Dictionary containing prompts (points, boxes, etc.)
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Set image for prediction
            if hasattr(self.image_predictor, 'set_image'):
                self.image_predictor.set_image(rgb_frame)
            
                # Prepare prompts
                point_coords = None
                point_labels = None
                box = None
                
                if prompts:
                    if 'points' in prompts:
                        point_coords = np.array(prompts['points'])
                        point_labels = np.array(prompts.get('point_labels', [1] * len(prompts['points'])))
                    
                    if 'box' in prompts:
                        box = np.array(prompts['box'])
                
                # Run prediction
                masks, scores, logits = self.image_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=box,
                    multimask_output=True
                )
                
                # Filter by confidence
                valid_masks = scores >= self.confidence_threshold
                
                return {
                    'masks': masks[valid_masks],
                    'scores': scores[valid_masks],
                    'logits': logits[valid_masks] if logits is not None else None,
                    'frame_shape': frame.shape[:2]
                }
            else:
                # Fallback for development - return dummy results
                return {
                    'masks': np.zeros((1, frame.shape[0], frame.shape[1]), dtype=bool),
                    'scores': np.array([0.5]),
                    'logits': None,
                    'frame_shape': frame.shape[:2]
                }
                
        except Exception as e:
            self.logger.error(f"Frame segmentation failed: {e}")
            return {
                'masks': np.array([]),
                'scores': np.array([]),
                'logits': None,
                'frame_shape': frame.shape[:2]
            }
    
    def segment_video_frame(
        self,
        frame_idx: int,
        prompts: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Segment a frame in video context using SAM2's video capabilities.
        
        Args:
            frame_idx: Frame index in the video
            prompts: Dictionary containing prompts
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            if not hasattr(self.video_predictor, 'add_new_points') or self.inference_state is None:
                # Fallback to image segmentation
                return {'masks': np.array([]), 'scores': np.array([]), 'object_ids': np.array([])}
            
            # Add prompts if provided
            if prompts and 'points' in prompts:
                obj_id = prompts.get('object_id', 0)
                points = np.array(prompts['points'])
                labels = np.array(prompts.get('point_labels', [1] * len(prompts['points'])))
                
                self.video_predictor.add_new_points(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels
                )
            
            # Propagate masks
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                self.inference_state
            ):
                video_segments[out_frame_idx] = {
                    'object_ids': out_obj_ids,
                    'mask_logits': out_mask_logits
                }
            
            # Get results for current frame
            if frame_idx in video_segments:
                frame_results = video_segments[frame_idx]
                masks = (frame_results['mask_logits'] > self.mask_threshold).cpu().numpy()
                scores = torch.sigmoid(frame_results['mask_logits']).max(dim=0)[0].cpu().numpy()
                
                return {
                    'masks': masks,
                    'scores': scores,
                    'object_ids': frame_results['object_ids'],
                    'mask_logits': frame_results['mask_logits']
                }
            else:
                return {'masks': np.array([]), 'scores': np.array([]), 'object_ids': np.array([])}
                
        except Exception as e:
            self.logger.error(f"Video frame segmentation failed: {e}")
            return {'masks': np.array([]), 'scores': np.array([]), 'object_ids': np.array([])}
    
    def generate_auto_prompts(self, frame: np.ndarray, num_points: int = 10) -> Dict:
        """
        Generate automatic prompts for segmentation.
        
        Args:
            frame: Input frame
            num_points: Number of points to generate
            
        Returns:
            Dictionary containing generated prompts
        """
        # Simple grid-based point generation
        h, w = frame.shape[:2]
        
        # Generate grid points
        points = []
        for i in range(int(np.sqrt(num_points))):
            for j in range(int(np.sqrt(num_points))):
                x = int(w * (j + 1) / (int(np.sqrt(num_points)) + 1))
                y = int(h * (i + 1) / (int(np.sqrt(num_points)) + 1))
                points.append([x, y])
        
        return {
            'points': points[:num_points],
            'point_labels': [1] * min(num_points, len(points))
        }
    
    def postprocess_masks(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        min_area: int = 100,
        max_area: int = 50000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Post-process segmentation masks.
        
        Args:
            masks: Segmentation masks
            scores: Confidence scores
            min_area: Minimum mask area
            max_area: Maximum mask area
            
        Returns:
            Filtered masks and scores
        """
        if len(masks) == 0:
            return masks, scores
        
        # Filter by area
        valid_indices = []
        for i, mask in enumerate(masks):
            area = np.sum(mask)
            if min_area <= area <= max_area:
                valid_indices.append(i)
        
        if valid_indices:
            return masks[valid_indices], scores[valid_indices]
        else:
            return np.array([]), np.array([])
    
    def visualize_masks(
        self,
        frame: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        alpha: float = 0.6,
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Visualize segmentation masks on frame.
        
        Args:
            frame: Input frame
            masks: Segmentation masks
            scores: Confidence scores
            alpha: Transparency of mask overlay
            colors: Colors for each mask
            
        Returns:
            Frame with mask overlay
        """
        if len(masks) == 0:
            return frame
        
        overlay = frame.copy()
        
        # Default colors
        if colors is None:
            colors = [
                (0, 255, 0),    # Green
                (0, 255, 255),  # Yellow
                (255, 165, 0),  # Orange
                (0, 165, 255),  # Red
                (255, 0, 255),  # Magenta
            ]
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            color = colors[i % len(colors)]
            
            # Apply mask with color
            mask_colored = np.zeros_like(frame)
            mask_colored[mask] = color
            
            # Blend with original frame
            overlay = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)
            
            # Add score text
            if np.any(mask):
                # Find mask centroid
                y_coords, x_coords = np.where(mask)
                if len(y_coords) > 0:
                    centroid_x = int(np.mean(x_coords))
                    centroid_y = int(np.mean(y_coords))
                    
                    cv2.putText(
                        overlay,
                        f"{score:.2f}",
                        (centroid_x, centroid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
        
        return overlay
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'inference_state') and self.inference_state is not None:
            try:
                if hasattr(self.video_predictor, 'reset_state'):
                    self.video_predictor.reset_state(self.inference_state)
            except:
                pass
        
        self.inference_state = None
        self.current_video_path = None
        self.frame_idx = 0
