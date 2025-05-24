"""
Optimized Elk Detector

This module provides a YOLO detector specifically optimized for maximum elk detection
in helicopter footage, prioritizing elk count accuracy over classification precision.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import torch
from .yolo_detector import YOLODetector


class OptimizedElkDetector(YOLODetector):
    """
    Optimized elk detector focused on maximum elk detection.
    
    This class extends YOLODetector with optimizations for:
    - Maximum elk detection (very low confidence thresholds)
    - Simplified bull/cow classification (no calves - calves counted as cows)
    - All detections labeled as "Elk" only
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.15,  # Very low for maximum detection
        iou_threshold: float = 0.4,  # Lower to avoid merging separate elk
        elk_size_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the optimized elk detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on
            confidence_threshold: Very low threshold for maximum elk detection
            iou_threshold: Lower IoU threshold to avoid merging elk
            elk_size_thresholds: Size thresholds for elk classification
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
        
        # Simplified elk size classification thresholds (optimized for smaller elk)
        self.elk_size_thresholds = elk_size_thresholds or {
            'cow': 1200,       # Smaller threshold - calves and cows grouped together
            'bull': float('inf')  # Only distinguish bulls from cows
        }
        
        # Motion compensation parameters
        self.prev_frame = None
        self.motion_threshold = 50
        
        # Colors for elk types (simplified)
        self.elk_colors = {
            'bull': (0, 255, 0),      # Green for bulls
            'cow': (0, 255, 255),     # Yellow for cows (includes calves)
            'unknown': (0, 165, 255)  # Red for unknown
        }
    
    def classify_elk_by_size(self, bbox: np.ndarray) -> str:
        """
        Classify elk type based on bounding box size (simplified).
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Elk type: 'bull' or 'cow' (calves classified as cows)
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        if area > self.elk_size_thresholds['cow']:
            return 'bull'
        else:
            return 'cow'  # Includes calves as requested
    
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
    
    def detect_elk(
        self,
        frame: np.ndarray,
        apply_motion_compensation: bool = True
    ) -> Dict:
        """
        Detect elk with optimized processing for maximum elk count.
        
        Args:
            frame: Input frame
            apply_motion_compensation: Whether to apply motion compensation
            
        Returns:
            Optimized detection results focused on elk count
        """
        
        # Apply motion compensation if requested
        processed_frame = frame
        motion_detected = False
        
        if apply_motion_compensation:
            processed_frame, motion_detected = self.compensate_for_motion(frame)
        
        # Estimate motion blur and adjust confidence if needed
        blur_score = self.estimate_motion_blur(processed_frame)
        adaptive_confidence = self.confidence_threshold
        
        # Be even more aggressive with low confidence for blurry frames
        if blur_score < 100:  # Blurry frame
            adaptive_confidence *= 0.6  # Even lower threshold for blurry frames
        
        # Run YOLO detection with very low confidence
        detections = self.detect(processed_frame, confidence=adaptive_confidence)
        
        # Process all detections as elk
        elk_detections = []
        
        for i in range(len(detections['boxes'])):
            bbox = detections['boxes'][i]
            confidence = detections['scores'][i]
            class_id = detections['class_ids'][i]
            class_name = detections['class_names'][i]
            
            # Classify elk by size (simplified)
            elk_type = self.classify_elk_by_size(bbox)
            
            # All detections are elk - boost confidence slightly
            elk_confidence = min(confidence * 1.1, 1.0)
            
            elk_detection = {
                'bbox': bbox,
                'confidence': confidence,
                'elk_confidence': elk_confidence,
                'class_id': class_id,
                'class_name': class_name,  # Original YOLO class for reference
                'elk_type': elk_type,
                'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            }
            
            elk_detections.append(elk_detection)
        
        return {
            'elk_detections': elk_detections,
            'motion_detected': motion_detected,
            'blur_score': blur_score,
            'frame_processed': processed_frame,
            'total_elk_count': len(elk_detections),
            'elk_types': {
                'bulls': len([d for d in elk_detections if d['elk_type'] == 'bull']),
                'cows': len([d for d in elk_detections if d['elk_type'] == 'cow']),
                'calves': 0  # Calves are counted as cows
            }
        }
    
    def get_elk_color(self, elk_type: str) -> Tuple[int, int, int]:
        """Get color for elk type visualization."""
        return self.elk_colors.get(elk_type, self.elk_colors['unknown'])
