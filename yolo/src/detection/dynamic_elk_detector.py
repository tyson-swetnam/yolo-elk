"""
Dynamic Elk Detector

This module provides an adaptive YOLO detector specifically designed for elk detection
in helicopter footage with dynamic size adaptation and movement-aware processing.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import torch
from collections import deque
from .yolo_detector import YOLODetector


class DynamicElkDetector(YOLODetector):
    """
    Dynamic elk detector that adapts to changing elk sizes and helicopter movement.
    
    Features:
    - Adaptive size thresholds based on frame content
    - Temporal tracking of elk across size changes
    - Movement-aware processing for helicopter footage
    - Multi-scale detection with dynamic classification
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.12,  # Very low base threshold
        iou_threshold: float = 0.3,  # Lower to avoid merging separate elk
        min_elk_size: int = 50,  # Minimum elk size in pixels
        history_length: int = 10  # Frames to keep in history
    ):
        """
        Initialize the dynamic elk detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on
            confidence_threshold: Base confidence threshold
            iou_threshold: IoU threshold for NMS
            min_elk_size: Minimum elk size in pixels
            history_length: Number of frames to keep in history
        """
        
        # Elk-relevant COCO classes only
        elk_classes = [17, 18, 19, 21]  # horse, sheep, cow, bear
        
        super().__init__(
            model_path=model_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            target_classes=elk_classes
        )
        
        # Dynamic parameters
        self.min_elk_size = min_elk_size
        self.history_length = history_length
        
        # Frame history for temporal analysis
        self.frame_history = deque(maxlen=history_length)
        self.size_history = deque(maxlen=history_length)
        self.elk_count_history = deque(maxlen=history_length)
        
        # Motion compensation parameters
        self.prev_frame = None
        self.motion_threshold = 30  # Lower threshold for more sensitive detection
        
        # Adaptive thresholds
        self.adaptive_confidence_range = (0.08, 0.25)  # Min/max confidence
        self.size_percentile_threshold = 0.6  # Top 40% are bulls
        
        # Colors for elk types
        self.elk_colors = {
            'bull': (0, 255, 0),      # Green for bulls
            'cow': (0, 255, 255),     # Yellow for cows
            'unknown': (0, 165, 255)  # Red for unknown
        }
    
    def estimate_helicopter_movement(self, frame: np.ndarray) -> Dict:
        """
        Estimate helicopter movement and distance changes.
        
        Args:
            frame: Current frame
            
        Returns:
            Dictionary with movement analysis
        """
        if self.prev_frame is None:
            self.prev_frame = frame.copy()
            return {'movement_detected': False, 'movement_magnitude': 0, 'direction': 'stable'}
        
        # Convert to grayscale for motion estimation
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
        
        movement_info = {'movement_detected': False, 'movement_magnitude': 0, 'direction': 'stable'}
        
        try:
            # Use ORB for feature detection
            orb = cv2.ORB_create(nfeatures=1000)
            kp1, des1 = orb.detectAndCompute(gray_prev, None)
            kp2, des2 = orb.detectAndCompute(gray_curr, None)
            
            if des1 is not None and des2 is not None and len(des1) > 20 and len(des2) > 20:
                # Match features
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 20:
                    # Extract matched points
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
                    
                    # Calculate movement vectors
                    movement_vectors = dst_pts - src_pts
                    avg_movement = np.mean(movement_vectors, axis=0)[0]
                    movement_magnitude = np.linalg.norm(avg_movement)
                    
                    movement_info['movement_magnitude'] = float(movement_magnitude)
                    
                    if movement_magnitude > self.motion_threshold:
                        movement_info['movement_detected'] = True
                        
                        # Determine movement direction (simplified)
                        if abs(avg_movement[0]) > abs(avg_movement[1]):
                            movement_info['direction'] = 'horizontal'
                        else:
                            movement_info['direction'] = 'vertical'
                            
                        # Estimate if helicopter is getting closer or farther
                        # This is a simplified heuristic based on feature density changes
                        feature_density_prev = len(kp1) / (gray_prev.shape[0] * gray_prev.shape[1])
                        feature_density_curr = len(kp2) / (gray_curr.shape[0] * gray_curr.shape[1])
                        
                        if feature_density_curr > feature_density_prev * 1.1:
                            movement_info['distance_change'] = 'closer'
                        elif feature_density_curr < feature_density_prev * 0.9:
                            movement_info['distance_change'] = 'farther'
                        else:
                            movement_info['distance_change'] = 'stable'
        
        except Exception:
            pass
        
        self.prev_frame = frame.copy()
        return movement_info
    
    def analyze_frame_elk_sizes(self, detections: List[Dict]) -> Dict:
        """
        Analyze elk sizes in current frame to determine dynamic thresholds.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Size analysis results
        """
        if not detections:
            return {
                'elk_sizes': [],
                'size_range': (0, 0),
                'bull_threshold': self.min_elk_size,
                'avg_size': 0,
                'size_distribution': []
            }
        
        # Extract elk sizes
        elk_sizes = [det['area'] for det in detections]
        elk_sizes.sort(reverse=True)  # Largest first
        
        # Calculate statistics
        min_size = min(elk_sizes)
        max_size = max(elk_sizes)
        avg_size = np.mean(elk_sizes)
        
        # Dynamic bull threshold based on size distribution
        if len(elk_sizes) > 1:
            # Bulls are the largest X% of elk in the frame
            bull_threshold_idx = max(1, int(len(elk_sizes) * self.size_percentile_threshold))
            bull_threshold = elk_sizes[bull_threshold_idx - 1]
        else:
            # Single elk - use historical data or minimum
            bull_threshold = max(self.min_elk_size * 2, avg_size)
        
        # Store in history
        self.size_history.append({
            'sizes': elk_sizes,
            'avg_size': avg_size,
            'bull_threshold': bull_threshold
        })
        
        return {
            'elk_sizes': elk_sizes,
            'size_range': (min_size, max_size),
            'bull_threshold': bull_threshold,
            'avg_size': avg_size,
            'size_distribution': elk_sizes
        }
    
    def get_adaptive_confidence(self, movement_info: Dict, frame_quality: float) -> float:
        """
        Calculate adaptive confidence threshold based on conditions.
        
        Args:
            movement_info: Movement analysis results
            frame_quality: Frame quality score
            
        Returns:
            Adaptive confidence threshold
        """
        base_confidence = self.confidence_threshold
        
        # Adjust for movement
        if movement_info['movement_detected']:
            movement_magnitude = movement_info['movement_magnitude']
            # Lower confidence for high movement (more blur)
            movement_factor = max(0.7, 1.0 - (movement_magnitude / 100.0))
            base_confidence *= movement_factor
        
        # Adjust for frame quality
        if frame_quality < 100:  # Blurry frame
            base_confidence *= 0.8
        
        # Adjust for distance changes
        if movement_info.get('distance_change') == 'farther':
            base_confidence *= 0.7  # Lower threshold for distant elk
        elif movement_info.get('distance_change') == 'closer':
            base_confidence *= 1.1  # Higher threshold for close elk
        
        # Clamp to reasonable range
        return np.clip(base_confidence, 
                      self.adaptive_confidence_range[0], 
                      self.adaptive_confidence_range[1])
    
    def classify_elk_dynamically(self, detections: List[Dict], size_analysis: Dict) -> List[Dict]:
        """
        Classify elk as bulls or cows based on dynamic size analysis.
        
        Args:
            detections: List of detection dictionaries
            size_analysis: Size analysis results
            
        Returns:
            Updated detections with elk classifications
        """
        bull_threshold = size_analysis['bull_threshold']
        
        for detection in detections:
            area = detection['area']
            
            # Dynamic classification based on relative size in frame
            if area >= bull_threshold:
                detection['elk_type'] = 'bull'
            else:
                detection['elk_type'] = 'cow'
            
            # Add size context
            detection['relative_size'] = area / size_analysis['avg_size'] if size_analysis['avg_size'] > 0 else 1.0
            detection['size_rank'] = size_analysis['elk_sizes'].index(area) + 1 if area in size_analysis['elk_sizes'] else len(size_analysis['elk_sizes'])
        
        return detections
    
    def detect_elk_dynamic(
        self,
        frame: np.ndarray,
        apply_motion_compensation: bool = True
    ) -> Dict:
        """
        Detect elk with dynamic adaptation to changing conditions.
        
        Args:
            frame: Input frame
            apply_motion_compensation: Whether to apply motion compensation
            
        Returns:
            Enhanced detection results with dynamic adaptation
        """
        
        # Analyze helicopter movement
        movement_info = self.estimate_helicopter_movement(frame)
        
        # Estimate frame quality
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Get adaptive confidence threshold
        adaptive_confidence = self.get_adaptive_confidence(movement_info, blur_score)
        
        # Run YOLO detection with adaptive confidence
        detections = self.detect(frame, confidence=adaptive_confidence)
        
        # Convert to elk detection format
        elk_detections = []
        for i in range(len(detections['boxes'])):
            bbox = detections['boxes'][i]
            confidence = detections['scores'][i]
            class_id = detections['class_ids'][i]
            class_name = detections['class_names'][i]
            
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Filter by minimum size
            if area >= self.min_elk_size:
                elk_detection = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                    'area': area,
                    'elk_type': 'unknown'  # Will be set by dynamic classification
                }
                elk_detections.append(elk_detection)
        
        # Analyze elk sizes in this frame
        size_analysis = self.analyze_frame_elk_sizes(elk_detections)
        
        # Classify elk dynamically
        elk_detections = self.classify_elk_dynamically(elk_detections, size_analysis)
        
        # Update elk count history
        elk_count = len(elk_detections)
        self.elk_count_history.append(elk_count)
        
        # Calculate elk type counts
        elk_types = {
            'bulls': len([d for d in elk_detections if d['elk_type'] == 'bull']),
            'cows': len([d for d in elk_detections if d['elk_type'] == 'cow']),
            'calves': 0  # Calves counted as cows
        }
        
        return {
            'elk_detections': elk_detections,
            'movement_info': movement_info,
            'blur_score': blur_score,
            'adaptive_confidence': adaptive_confidence,
            'size_analysis': size_analysis,
            'total_elk_count': elk_count,
            'elk_types': elk_types,
            'frame_processed': frame
        }
    
    def get_elk_color(self, elk_type: str) -> Tuple[int, int, int]:
        """Get color for elk type visualization."""
        return self.elk_colors.get(elk_type, self.elk_colors['unknown'])
    
    def get_detection_summary(self) -> Dict:
        """Get summary of detection history."""
        if not self.elk_count_history:
            return {'max_elk': 0, 'avg_elk': 0, 'total_frames': 0}
        
        return {
            'max_elk': max(self.elk_count_history),
            'avg_elk': np.mean(self.elk_count_history),
            'total_frames': len(self.elk_count_history),
            'elk_count_trend': list(self.elk_count_history)
        }
