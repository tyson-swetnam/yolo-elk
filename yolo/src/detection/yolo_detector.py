"""
YOLO Object Detector

This module provides a wrapper around Ultralytics YOLO for object detection
with enhanced functionality for tracking applications.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from ultralytics import YOLO
import torch


class YOLODetector:
    """
    YOLO Object Detector for tracking applications.
    
    This class provides an interface to YOLO models with additional
    functionality for filtering detections and preparing data for tracking.
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.7,
        target_classes: Optional[List[int]] = None
    ):
        """
        Initialize the YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ('cpu', 'cuda', 'auto')
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            target_classes: List of class IDs to detect (None for all classes)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.target_classes = target_classes
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Get class names
        self.class_names = self.model.names
        
    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None
    ) -> Dict:
        """
        Detect objects in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence: Override confidence threshold for this detection
            iou: Override IoU threshold for this detection
            
        Returns:
            Dictionary containing detection results with keys:
            - 'boxes': Bounding boxes as [x1, y1, x2, y2]
            - 'scores': Confidence scores
            - 'class_ids': Class IDs
            - 'class_names': Class names
        """
        conf = confidence if confidence is not None else self.confidence_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold
        
        # Run inference
        results = self.model(
            image,
            conf=conf,
            iou=iou_thresh,
            classes=self.target_classes,
            verbose=False
        )
        
        # Extract results
        result = results[0]
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            scores = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            class_names = [self.class_names[cls_id] for cls_id in class_ids]
        else:
            boxes = np.empty((0, 4))
            scores = np.empty(0)
            class_ids = np.empty(0, dtype=int)
            class_names = []
            
        return {
            'boxes': boxes,
            'scores': scores,
            'class_ids': class_ids,
            'class_names': class_names
        }
    
    def detect_and_track_format(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
        iou: Optional[float] = None
    ) -> np.ndarray:
        """
        Detect objects and return in format suitable for tracking.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence: Override confidence threshold
            iou: Override IoU threshold
            
        Returns:
            Array of detections in format [x1, y1, x2, y2, score, class_id]
        """
        detections = self.detect(image, confidence, iou)
        
        if len(detections['boxes']) == 0:
            return np.empty((0, 6))
            
        # Combine into tracking format
        track_detections = np.column_stack([
            detections['boxes'],
            detections['scores'],
            detections['class_ids']
        ])
        
        return track_detections
    
    def filter_by_class(
        self,
        detections: Dict,
        target_classes: List[Union[int, str]]
    ) -> Dict:
        """
        Filter detections by class.
        
        Args:
            detections: Detection results from detect()
            target_classes: List of class IDs or names to keep
            
        Returns:
            Filtered detection results
        """
        if len(detections['boxes']) == 0:
            return detections
            
        # Convert class names to IDs if necessary
        if isinstance(target_classes[0], str):
            class_name_to_id = {name: id for id, name in self.class_names.items()}
            target_class_ids = [class_name_to_id[name] for name in target_classes if name in class_name_to_id]
        else:
            target_class_ids = target_classes
            
        # Filter
        mask = np.isin(detections['class_ids'], target_class_ids)
        
        return {
            'boxes': detections['boxes'][mask],
            'scores': detections['scores'][mask],
            'class_ids': detections['class_ids'][mask],
            'class_names': [detections['class_names'][i] for i in range(len(mask)) if mask[i]]
        }
    
    def filter_by_area(
        self,
        detections: Dict,
        min_area: float = 0,
        max_area: float = float('inf')
    ) -> Dict:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: Detection results from detect()
            min_area: Minimum bounding box area
            max_area: Maximum bounding box area
            
        Returns:
            Filtered detection results
        """
        if len(detections['boxes']) == 0:
            return detections
            
        boxes = detections['boxes']
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        mask = (areas >= min_area) & (areas <= max_area)
        
        return {
            'boxes': detections['boxes'][mask],
            'scores': detections['scores'][mask],
            'class_ids': detections['class_ids'][mask],
            'class_names': [detections['class_names'][i] for i in range(len(mask)) if mask[i]]
        }
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path,
            'device': self.device,
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'target_classes': self.target_classes
        }
