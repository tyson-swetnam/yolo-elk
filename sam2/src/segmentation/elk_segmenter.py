"""
Elk-Specific SAM2 Segmenter

This module provides elk-specific segmentation capabilities using SAM2,
including elk classification, motion compensation, and feature analysis.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import logging
from .sam2_segmenter import SAM2Segmenter


class ElkSegmenter(SAM2Segmenter):
    """
    Elk-specific segmenter with enhanced capabilities for helicopter footage.
    
    This class extends SAM2Segmenter with elk-specific features including:
    - Elk size classification (bull/cow/calf)
    - Motion compensation for helicopter footage
    - Elk-specific visual feature analysis
    - Intelligent prompting for elk detection
    """
    
    def __init__(
        self,
        model_checkpoint: str = "sam2_hiera_large.pt",
        model_config: str = "sam2_hiera_l.yaml",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        mask_threshold: float = 0.0,
        elk_size_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the elk segmenter.
        
        Args:
            model_checkpoint: Path to SAM2 model checkpoint
            model_config: Path to SAM2 model configuration
            device: Device to run inference on
            confidence_threshold: Minimum confidence for segmentations
            mask_threshold: Threshold for mask binarization
            elk_size_thresholds: Size thresholds for elk classification
        """
        
        super().__init__(
            model_checkpoint=model_checkpoint,
            model_config=model_config,
            device=device,
            confidence_threshold=confidence_threshold,
            mask_threshold=mask_threshold
        )
        
        # Elk size classification thresholds (based on mask area)
        self.elk_size_thresholds = elk_size_thresholds or {
            'calf': 2000,      # Small elk (calves)
            'cow': 8000,       # Medium elk (cows)
            'bull': float('inf')  # Large elk (bulls)
        }
        
        # Motion compensation parameters
        self.prev_frame = None
        self.motion_threshold = 30  # Pixels of motion to trigger compensation
        
        # Colors for different elk types
        self.elk_colors = {
            'bull': (0, 255, 0),      # Green for bulls
            'cow': (0, 255, 255),     # Yellow for cows  
            'calf': (255, 165, 0),    # Orange for calves
            'unknown': (0, 165, 255)  # Red for unknown
        }
        
        # Elk detection parameters
        self.min_elk_area = 500
        self.max_elk_area = 50000
        self.elk_aspect_ratio_range = (0.8, 3.0)
        
        # Feature analysis settings
        self.elk_color_ranges = {
            'brown_lower': np.array([10, 50, 20]),
            'brown_upper': np.array([20, 255, 200]),
            'tan_lower': np.array([15, 30, 100]),
            'tan_upper': np.array([35, 150, 255])
        }
        self.white_rump_threshold = 0.15
    
    def classify_elk_by_size(self, mask: np.ndarray) -> str:
        """
        Classify elk type based on mask area.
        
        Args:
            mask: Segmentation mask
            
        Returns:
            Elk type: 'bull', 'cow', 'calf', or 'unknown'
        """
        area = np.sum(mask)
        
        if area < self.elk_size_thresholds['calf']:
            return 'calf'
        elif area < self.elk_size_thresholds['cow']:
            return 'cow'
        else:
            return 'bull'
    
    def estimate_motion_blur(self, frame: np.ndarray) -> float:
        """
        Estimate motion blur in the frame using Laplacian variance.
        
        Args:
            frame: Input frame
            
        Returns:
            Blur score (lower = more blurry)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def compensate_for_motion(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Apply motion compensation for helicopter movement.
        
        Args:
            frame: Current frame
            
        Returns:
            Tuple of (compensated_frame, motion_detected)
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return frame, False
        
        # Convert to grayscale for motion estimation
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect features and estimate motion
        try:
            # Use ORB for feature detection
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(gray_prev, None)
            kp2, des2 = orb.detectAndCompute(gray_curr, None)
            
            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 10:
                    # Extract matched points
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:20]]).reshape(-1, 1, 2)
                    
                    # Estimate homography
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        # Check if significant motion detected
                        motion_magnitude = np.linalg.norm(M[:2, 2])
                        
                        if motion_magnitude > self.motion_threshold:
                            # Apply inverse transformation to stabilize
                            h, w = frame.shape[:2]
                            stabilized = cv2.warpPerspective(frame, np.linalg.inv(M), (w, h))
                            self.prev_frame = frame.copy()
                            return stabilized, True
            
        except Exception:
            # If motion compensation fails, return original frame
            pass
        
        self.prev_frame = frame.copy()
        return frame, False
    
    def analyze_elk_features(self, frame: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Analyze elk-specific visual features in the segmented region.
        
        Args:
            frame: Input frame
            mask: Segmentation mask
            
        Returns:
            Dictionary with feature analysis results
        """
        if not np.any(mask):
            return {
                'has_elk_colors': False, 
                'aspect_ratio': 0, 
                'white_rump_score': 0,
                'elk_color_ratio': 0,
                'compactness': 0
            }
        
        # Extract region of interest using mask
        masked_frame = frame.copy()
        masked_frame[~mask] = 0
        
        # Get bounding box of mask
        y_coords, x_coords = np.where(mask)
        x1, x2 = np.min(x_coords), np.max(x_coords)
        y1, y2 = np.min(y_coords), np.max(y_coords)
        
        roi = frame[y1:y2+1, x1:x2+1]
        roi_mask = mask[y1:y2+1, x1:x2+1]
        
        if roi.size == 0:
            return {
                'has_elk_colors': False, 
                'aspect_ratio': 0, 
                'white_rump_score': 0,
                'elk_color_ratio': 0,
                'compactness': 0
            }
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Check for elk colors
        brown_mask = cv2.inRange(hsv, self.elk_color_ranges['brown_lower'], self.elk_color_ranges['brown_upper'])
        tan_mask = cv2.inRange(hsv, self.elk_color_ranges['tan_lower'], self.elk_color_ranges['tan_upper'])
        elk_color_mask = cv2.bitwise_or(brown_mask, tan_mask)
        
        # Apply ROI mask to color analysis
        elk_color_mask = cv2.bitwise_and(elk_color_mask, roi_mask.astype(np.uint8) * 255)
        
        elk_color_ratio = np.sum(elk_color_mask > 0) / np.sum(roi_mask) if np.sum(roi_mask) > 0 else 0
        has_elk_colors = elk_color_ratio > 0.1
        
        # Calculate aspect ratio
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        aspect_ratio = width / height if height > 0 else 0
        
        # Look for white rump (bottom portion of mask)
        mask_height = y2 - y1 + 1
        bottom_start = y1 + int(mask_height * 0.7)
        bottom_mask = mask[bottom_start:y2+1, x1:x2+1] if bottom_start < y2 else np.array([])
        
        white_rump_score = 0
        if bottom_mask.size > 0:
            bottom_roi = frame[bottom_start:y2+1, x1:x2+1]
            gray_bottom = cv2.cvtColor(bottom_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply mask to bottom region
            gray_bottom_masked = gray_bottom * bottom_mask.astype(np.uint8)
            
            if np.sum(bottom_mask) > 0:
                white_pixels = np.sum((gray_bottom_masked > 200) & bottom_mask)
                white_rump_score = white_pixels / np.sum(bottom_mask)
        
        # Calculate compactness (how circular the shape is)
        area = np.sum(mask)
        perimeter = cv2.arcLength(cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0], True)
        compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        return {
            'has_elk_colors': has_elk_colors,
            'elk_color_ratio': elk_color_ratio,
            'aspect_ratio': aspect_ratio,
            'white_rump_score': white_rump_score,
            'compactness': compactness,
            'area': area
        }
    
    def generate_elk_prompts(self, frame: np.ndarray, method: str = "adaptive") -> Dict:
        """
        Generate elk-specific prompts for segmentation.
        
        Args:
            frame: Input frame
            method: Prompting method ('adaptive', 'grid', 'edge_based')
            
        Returns:
            Dictionary containing generated prompts
        """
        h, w = frame.shape[:2]
        
        if method == "adaptive":
            # Use adaptive prompting based on frame content
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find potential elk regions using simple blob detection
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use adaptive threshold to find potential objects
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            points = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_elk_area <= area <= self.max_elk_area:
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        points.append([cx, cy])
            
            # If no good contours found, fall back to grid
            if len(points) < 3:
                return self.generate_auto_prompts(frame, num_points=9)
            
            return {
                'points': points[:10],  # Limit to 10 points
                'point_labels': [1] * min(10, len(points))
            }
            
        elif method == "edge_based":
            # Use edge detection to find potential elk boundaries
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            points = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Minimum area for edge-based detection
                    # Get multiple points along the contour
                    contour_points = contour.reshape(-1, 2)
                    # Sample every 10th point
                    sampled_points = contour_points[::max(1, len(contour_points)//5)]
                    points.extend(sampled_points.tolist())
            
            if len(points) < 3:
                return self.generate_auto_prompts(frame, num_points=9)
            
            return {
                'points': points[:15],  # Limit to 15 points
                'point_labels': [1] * min(15, len(points))
            }
            
        else:  # grid method
            return self.generate_auto_prompts(frame, num_points=9)
    
    def segment_elk(
        self,
        frame: np.ndarray,
        apply_motion_compensation: bool = True,
        analyze_features: bool = True,
        prompting_method: str = "adaptive"
    ) -> Dict:
        """
        Segment elk with enhanced processing for helicopter footage.
        
        Args:
            frame: Input frame
            apply_motion_compensation: Whether to apply motion compensation
            analyze_features: Whether to analyze elk-specific features
            prompting_method: Method for generating prompts
            
        Returns:
            Enhanced segmentation results with elk classification
        """
        
        # Apply motion compensation if requested
        processed_frame = frame
        motion_detected = False
        
        if apply_motion_compensation:
            processed_frame, motion_detected = self.compensate_for_motion(frame)
        
        # Estimate motion blur and adjust confidence if needed
        blur_score = self.estimate_motion_blur(processed_frame)
        adaptive_confidence = self.confidence_threshold
        
        if blur_score < 100:  # Blurry frame
            adaptive_confidence *= 0.7  # Lower threshold for blurry frames
        
        # Generate elk-specific prompts
        prompts = self.generate_elk_prompts(processed_frame, method=prompting_method)
        
        # Run SAM2 segmentation
        segmentation_results = self.segment_frame(processed_frame, prompts=prompts)
        
        # Enhanced elk processing
        elk_segmentations = []
        
        if len(segmentation_results['masks']) > 0:
            masks = segmentation_results['masks']
            scores = segmentation_results['scores']
            
            # Post-process masks for elk-specific criteria
            filtered_masks, filtered_scores = self.postprocess_masks(
                masks, scores, 
                min_area=self.min_elk_area, 
                max_area=self.max_elk_area
            )
            
            for i, (mask, score) in enumerate(zip(filtered_masks, filtered_scores)):
                # Classify elk by size
                elk_type = self.classify_elk_by_size(mask)
                
                # Analyze elk features if requested
                features = {}
                if analyze_features:
                    features = self.analyze_elk_features(frame, mask)
                
                # Calculate elk confidence score
                elk_confidence = score
                
                # Boost confidence for elk-like features
                if features.get('has_elk_colors', False):
                    elk_confidence *= 1.2
                
                aspect_ratio = features.get('aspect_ratio', 0)
                if self.elk_aspect_ratio_range[0] <= aspect_ratio <= self.elk_aspect_ratio_range[1]:
                    elk_confidence *= 1.1
                
                if features.get('white_rump_score', 0) > self.white_rump_threshold:
                    elk_confidence *= 1.15
                
                # Boost for good compactness (elk-like shape)
                if features.get('compactness', 0) > 0.3:
                    elk_confidence *= 1.05
                
                elk_confidence = min(elk_confidence, 1.0)  # Cap at 1.0
                
                elk_segmentation = {
                    'mask': mask,
                    'confidence': score,
                    'elk_confidence': elk_confidence,
                    'elk_type': elk_type,
                    'features': features,
                    'area': np.sum(mask),
                    'centroid': self._get_mask_centroid(mask)
                }
                
                elk_segmentations.append(elk_segmentation)
        
        return {
            'elk_segmentations': elk_segmentations,
            'motion_detected': motion_detected,
            'blur_score': blur_score,
            'frame_processed': processed_frame,
            'total_elk_count': len(elk_segmentations),
            'elk_types': {
                'bulls': len([s for s in elk_segmentations if s['elk_type'] == 'bull']),
                'cows': len([s for s in elk_segmentations if s['elk_type'] == 'cow']),
                'calves': len([s for s in elk_segmentations if s['elk_type'] == 'calf'])
            },
            'prompts_used': prompts
        }
    
    def _get_mask_centroid(self, mask: np.ndarray) -> Tuple[int, int]:
        """Get centroid of mask."""
        if not np.any(mask):
            return (0, 0)
        
        y_coords, x_coords = np.where(mask)
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        return (centroid_x, centroid_y)
    
    def get_elk_color(self, elk_type: str) -> Tuple[int, int, int]:
        """Get color for elk type visualization."""
        return self.elk_colors.get(elk_type, self.elk_colors['unknown'])
    
    def visualize_elk_segmentations(
        self,
        frame: np.ndarray,
        elk_results: Dict,
        show_labels: bool = True,
        show_confidence: bool = True,
        alpha: float = 0.6
    ) -> np.ndarray:
        """
        Visualize elk segmentations on frame.
        
        Args:
            frame: Input frame
            elk_results: Results from elk segmentation
            show_labels: Whether to show elk type labels
            show_confidence: Whether to show confidence scores
            alpha: Transparency of mask overlay
            
        Returns:
            Frame with elk segmentations visualized
        """
        if not elk_results['elk_segmentations']:
            return frame
        
        overlay = frame.copy()
        
        for elk_seg in elk_results['elk_segmentations']:
            mask = elk_seg['mask']
            elk_type = elk_seg['elk_type']
            confidence = elk_seg['elk_confidence']
            centroid = elk_seg['centroid']
            
            # Get color for elk type
            color = self.get_elk_color(elk_type)
            
            # Apply mask with color
            mask_colored = np.zeros_like(frame)
            mask_colored[mask] = color
            
            # Blend with original frame
            overlay = cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0)
            
            # Add labels if requested
            if show_labels or show_confidence:
                label_parts = []
                if show_labels:
                    label_parts.append(elk_type.upper())
                if show_confidence:
                    label_parts.append(f"{confidence:.2f}")
                
                label = ": ".join(label_parts)
                
                cv2.putText(
                    overlay,
                    label,
                    centroid,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
        
        return overlay
