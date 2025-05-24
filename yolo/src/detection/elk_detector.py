"""
Elk-Specific Object Detector

This module provides an enhanced YOLO detector specifically optimized for elk detection
in helicopter footage, with motion compensation and elk classification capabilities.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import torch
from .yolo_detector import YOLODetector


class ElkDetector(YOLODetector):
    """
    Elk-specific detector with enhanced capabilities for helicopter footage.
    
    This class extends YOLODetector with elk-specific features including:
    - Elk size classification (bull/cow/calf)
    - Motion compensation for helicopter footage
    - Elk-specific visual feature analysis
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.25,  # Lower for motion blur
        iou_threshold: float = 0.5,
        elk_size_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the elk detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on
            confidence_threshold: Lower threshold for motion-blurred elk
            iou_threshold: IoU threshold for NMS
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
        
        # Elk size classification thresholds (based on bounding box area)
        self.elk_size_thresholds = elk_size_thresholds or {
            'calf': 1500,      # Small elk (calves)
            'cow': 4000,       # Medium elk (cows)
            'bull': float('inf')  # Large elk (bulls)
        }
        
        # Motion compensation parameters
        self.prev_frame = None
        self.motion_threshold = 50  # Pixels of motion to trigger compensation
        
        # Elk-specific keywords for classification
        self.elk_keywords = ['horse', 'cow', 'sheep', 'bear']
        
        # Colors for different elk types
        self.elk_colors = {
            'bull': (0, 255, 0),      # Green for bulls
            'cow': (0, 255, 255),     # Yellow for cows  
            'calf': (255, 165, 0),    # Orange for calves
            'unknown': (0, 165, 255)  # Red for unknown
        }
    
    def classify_elk_by_size(self, bbox: np.ndarray) -> str:
        """
        Classify elk type based on bounding box size.
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Elk type: 'bull', 'cow', 'calf', or 'unknown'
        """
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
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
    
    def analyze_elk_features(self, frame: np.ndarray, bbox: np.ndarray) -> Dict:
        """
        Analyze elk-specific visual features in the detection region.
        
        Args:
            frame: Input frame
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary with feature analysis results
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Extract region of interest
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return {'has_elk_colors': False, 'aspect_ratio': 0, 'white_rump_score': 0}
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define elk color ranges (brown/tan)
        brown_lower = np.array([10, 50, 20])
        brown_upper = np.array([20, 255, 200])
        tan_lower = np.array([15, 30, 100])
        tan_upper = np.array([35, 150, 255])
        
        # Check for elk colors
        brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
        tan_mask = cv2.inRange(hsv, tan_lower, tan_upper)
        elk_color_mask = cv2.bitwise_or(brown_mask, tan_mask)
        
        elk_color_ratio = np.sum(elk_color_mask > 0) / elk_color_mask.size
        has_elk_colors = elk_color_ratio > 0.1
        
        # Calculate aspect ratio (elk are typically longer than tall)
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0
        
        # Look for white rump (bottom portion of detection)
        bottom_third = roi[int(height * 0.7):, :]
        if bottom_third.size > 0:
            # Convert to grayscale and look for bright regions
            gray_bottom = cv2.cvtColor(bottom_third, cv2.COLOR_BGR2GRAY)
            white_pixels = np.sum(gray_bottom > 200)
            white_rump_score = white_pixels / gray_bottom.size
        else:
            white_rump_score = 0
        
        return {
            'has_elk_colors': has_elk_colors,
            'elk_color_ratio': elk_color_ratio,
            'aspect_ratio': aspect_ratio,
            'white_rump_score': white_rump_score
        }
    
    def detect_elk(
        self,
        frame: np.ndarray,
        apply_motion_compensation: bool = True,
        analyze_features: bool = True
    ) -> Dict:
        """
        Detect elk with enhanced processing for helicopter footage.
        
        Args:
            frame: Input frame
            apply_motion_compensation: Whether to apply motion compensation
            analyze_features: Whether to analyze elk-specific features
            
        Returns:
            Enhanced detection results with elk classification
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
        
        # Run YOLO detection
        detections = self.detect(processed_frame, confidence=adaptive_confidence)
        
        # Enhanced elk processing
        elk_detections = []
        
        for i in range(len(detections['boxes'])):
            bbox = detections['boxes'][i]
            confidence = detections['scores'][i]
            class_id = detections['class_ids'][i]
            class_name = detections['class_names'][i]
            
            # Classify elk by size
            elk_type = self.classify_elk_by_size(bbox)
            
            # Analyze elk features if requested
            features = {}
            if analyze_features:
                features = self.analyze_elk_features(frame, bbox)
            
            # Calculate elk confidence score
            elk_confidence = confidence
            
            # Boost confidence for elk-like features
            if features.get('has_elk_colors', False):
                elk_confidence *= 1.2
            
            if 1.2 <= features.get('aspect_ratio', 0) <= 2.5:  # Elk-like aspect ratio
                elk_confidence *= 1.1
            
            if features.get('white_rump_score', 0) > 0.1:
                elk_confidence *= 1.15
            
            elk_confidence = min(elk_confidence, 1.0)  # Cap at 1.0
            
            elk_detection = {
                'bbox': bbox,
                'confidence': confidence,
                'elk_confidence': elk_confidence,
                'class_id': class_id,
                'class_name': class_name,
                'elk_type': elk_type,
                'features': features,
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
                'calves': len([d for d in elk_detections if d['elk_type'] == 'calf'])
            }
        }
    
    def get_elk_color(self, elk_type: str) -> Tuple[int, int, int]:
        """Get color for elk type visualization."""
        return self.elk_colors.get(elk_type, self.elk_colors['unknown'])
