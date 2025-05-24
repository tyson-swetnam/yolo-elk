"""
Norfair-based Mask Tracker for SAM2

This module provides a wrapper around the Norfair tracking library
for multi-object tracking with SAM2 segmentation masks.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
import cv2
import norfair
from norfair import Detection, Tracker, draw_tracked_objects


class NorfairTracker:
    """
    Norfair-based multi-object tracker adapted for segmentation masks.
    
    This class provides an interface to Norfair tracking with mask-based
    distance functions and tracking parameters optimized for SAM2 segmentations.
    """
    
    def __init__(
        self,
        distance_function: str = "iou",
        distance_threshold: float = 0.3,
        hit_counter_max: int = 15,
        initialization_delay: int = 3,
        pointwise_hit_counter_max: int = 4,
        detection_threshold: float = 0.1,
        past_detections_length: int = 4,
        reid_distance_function: Optional[Callable] = None,
        reid_distance_threshold: float = 0.3
    ):
        """
        Initialize the Norfair tracker for masks.
        
        Args:
            distance_function: Distance function type ('euclidean', 'iou', 'centroid', 'mask_iou')
            distance_threshold: Maximum distance for association
            hit_counter_max: Maximum frames to keep track without detection
            initialization_delay: Frames to wait before confirming track
            pointwise_hit_counter_max: Max frames for point-wise tracking
            detection_threshold: Minimum detection confidence
            past_detections_length: Number of past detections to store
            reid_distance_function: Re-identification distance function
            reid_distance_threshold: Re-ID distance threshold
        """
        self.distance_threshold = distance_threshold
        self.detection_threshold = detection_threshold
        
        # Set up distance function
        self.distance_function_name = distance_function
        if distance_function == "euclidean":
            self.distance_func = self._euclidean_distance
        elif distance_function == "iou":
            self.distance_func = self._iou_distance
        elif distance_function == "centroid":
            self.distance_func = self._centroid_distance
        elif distance_function == "mask_iou":
            self.distance_func = self._mask_iou_distance
        else:
            raise ValueError(f"Unknown distance function: {distance_function}")
        
        # Initialize tracker
        self.tracker = Tracker(
            distance_function=self.distance_func,
            distance_threshold=distance_threshold,
            hit_counter_max=hit_counter_max,
            initialization_delay=initialization_delay,
            pointwise_hit_counter_max=pointwise_hit_counter_max,
            past_detections_length=past_detections_length,
            reid_distance_function=reid_distance_function,
            reid_distance_threshold=reid_distance_threshold
        )
        
        self.frame_count = 0
        
    def update(
        self,
        segmentations: List[Dict],
        frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Update tracker with new segmentations.
        
        Args:
            segmentations: List of segmentation dictionaries with masks and metadata
            frame: Optional frame for visualization
            
        Returns:
            List of tracked objects with track information
        """
        self.frame_count += 1
        
        # Convert segmentations to Norfair format
        norfair_detections = self._convert_segmentations(segmentations)
        
        # Update tracker
        tracked_objects = self.tracker.update(
            detections=norfair_detections,
            period=1
        )
        
        # Convert back to our format
        tracks = self._convert_tracked_objects(tracked_objects)
        
        return tracks
    
    def _convert_segmentations(self, segmentations: List[Dict]) -> List[Detection]:
        """
        Convert segmentation list to Norfair Detection objects.
        
        Args:
            segmentations: List of segmentation dictionaries
            
        Returns:
            List of Norfair Detection objects
        """
        norfair_detections = []
        
        for seg in segmentations:
            confidence = seg.get('elk_confidence', seg.get('confidence', 0.0))
            
            if confidence >= self.detection_threshold:
                mask = seg['mask']
                
                # Get mask properties
                centroid = seg.get('centroid', self._get_mask_centroid(mask))
                bbox = self._mask_to_bbox(mask)
                area = seg.get('area', np.sum(mask))
                
                # Create detection points (centroid + bbox corners for better tracking)
                x1, y1, x2, y2 = bbox
                points = np.array([
                    [centroid[0], centroid[1]],  # Centroid
                    [x1, y1],                    # Top-left
                    [x2, y2],                    # Bottom-right
                    [(x1 + x2) / 2, y1],        # Top-center
                    [(x1 + x2) / 2, y2]         # Bottom-center
                ])
                
                detection = Detection(
                    points=points,
                    scores=np.array([confidence] * len(points)),
                    data={
                        'mask': mask,
                        'bbox': bbox,
                        'confidence': confidence,
                        'elk_confidence': confidence,
                        'elk_type': seg.get('elk_type', 'unknown'),
                        'features': seg.get('features', {}),
                        'area': area,
                        'centroid': centroid
                    }
                )
                
                norfair_detections.append(detection)
        
        return norfair_detections
    
    def _convert_tracked_objects(self, tracked_objects: List) -> List[Dict]:
        """
        Convert Norfair tracked objects to our format.
        
        Args:
            tracked_objects: List of Norfair TrackedObject instances
            
        Returns:
            List of track dictionaries
        """
        tracks = []
        
        for obj in tracked_objects:
            if obj.last_detection is not None:
                # Get the latest detection data
                detection_data = obj.last_detection.data
                
                track_info = {
                    'track_id': obj.id,
                    'mask': detection_data.get('mask'),
                    'bbox': detection_data.get('bbox', [0, 0, 0, 0]),
                    'confidence': detection_data.get('confidence', 0.0),
                    'elk_confidence': detection_data.get('elk_confidence', 0.0),
                    'elk_type': detection_data.get('elk_type', 'unknown'),
                    'features': detection_data.get('features', {}),
                    'area': detection_data.get('area', 0),
                    'age': obj.age,
                    'hit_counter': obj.hit_counter,
                    'last_distance': obj.last_distance,
                    'is_confirmed': obj.age >= self.tracker.initialization_delay,
                    'centroid': detection_data.get('centroid', [0, 0]),
                    'estimated_position': obj.estimate.tolist() if obj.estimate is not None else [0, 0],
                    'velocity': obj.velocity.tolist() if obj.velocity is not None else [0, 0],
                    'frame_count': self.frame_count
                }
                
                tracks.append(track_info)
        
        return tracks
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        draw_masks: bool = True,
        draw_labels: bool = True,
        draw_ids: bool = True,
        draw_trails: bool = False,
        mask_alpha: float = 0.6
    ) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            tracks: List of track dictionaries
            draw_masks: Whether to draw segmentation masks
            draw_labels: Whether to draw elk type labels
            draw_ids: Whether to draw track IDs
            draw_trails: Whether to draw tracking trails
            mask_alpha: Transparency of mask overlay
            
        Returns:
            Frame with tracking visualization
        """
        overlay = frame.copy()
        
        # Colors for different elk types
        elk_colors = {
            'bull': (0, 255, 0),      # Green
            'cow': (0, 255, 255),     # Yellow
            'calf': (255, 165, 0),    # Orange
            'unknown': (0, 165, 255)  # Red
        }
        
        for track in tracks:
            if not track['is_confirmed']:
                continue
                
            mask = track.get('mask')
            elk_type = track.get('elk_type', 'unknown')
            track_id = track['track_id']
            confidence = track.get('elk_confidence', 0.0)
            centroid = track.get('centroid', [0, 0])
            
            color = elk_colors.get(elk_type, elk_colors['unknown'])
            
            # Draw mask if available
            if draw_masks and mask is not None:
                mask_colored = np.zeros_like(frame)
                mask_colored[mask] = color
                overlay = cv2.addWeighted(overlay, 1 - mask_alpha, mask_colored, mask_alpha, 0)
            
            # Draw bounding box
            bbox = track.get('bbox', [0, 0, 0, 0])
            if bbox != [0, 0, 0, 0]:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw labels and IDs
            if draw_labels or draw_ids:
                label_parts = []
                if draw_ids:
                    label_parts.append(f"ID:{track_id}")
                if draw_labels:
                    label_parts.append(f"{elk_type.upper()}")
                    label_parts.append(f"{confidence:.2f}")
                
                label = " ".join(label_parts)
                
                # Position label at centroid
                label_pos = (int(centroid[0]), int(centroid[1]))
                
                # Draw label background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                
                cv2.rectangle(
                    overlay,
                    (label_pos[0] - 5, label_pos[1] - text_height - 5),
                    (label_pos[0] + text_width + 5, label_pos[1] + baseline + 5),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    overlay,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2
                )
            
            # Draw trails if requested
            if draw_trails and hasattr(track, 'trail_points'):
                trail_points = track.get('trail_points', [])
                if len(trail_points) > 1:
                    for i in range(1, len(trail_points)):
                        cv2.line(
                            overlay,
                            tuple(map(int, trail_points[i-1])),
                            tuple(map(int, trail_points[i])),
                            color,
                            2
                        )
        
        return overlay
    
    def get_track_count(self) -> int:
        """Get current number of active tracks."""
        return len([obj for obj in self.tracker.tracked_objects if obj.hit_counter > 0])
    
    def reset(self):
        """Reset the tracker."""
        self.tracker = Tracker(
            distance_function=self.distance_func,
            distance_threshold=self.distance_threshold,
            hit_counter_max=self.tracker.hit_counter_max,
            initialization_delay=self.tracker.initialization_delay,
            pointwise_hit_counter_max=self.tracker.pointwise_hit_counter_max,
            past_detections_length=self.tracker.past_detections_length,
            reid_distance_function=self.tracker.reid_distance_function,
            reid_distance_threshold=self.tracker.reid_distance_threshold
        )
        self.frame_count = 0
    
    def _euclidean_distance(self, tracked_object, detection):
        """Calculate Euclidean distance between tracked object and detection."""
        if tracked_object.estimate is None:
            return float('inf')
        
        # Get center points
        track_center = tracked_object.estimate[:2]  # x, y
        det_center = detection.points[0]  # First point is centroid
        
        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((track_center - det_center) ** 2))
        return distance
    
    def _iou_distance(self, tracked_object, detection):
        """Calculate IoU-based distance between tracked object and detection."""
        if tracked_object.last_detection is None:
            return float('inf')
        
        # Get bounding boxes
        track_bbox = tracked_object.last_detection.data['bbox']
        det_bbox = detection.data['bbox']
        
        # Calculate IoU
        iou = self._calculate_bbox_iou(track_bbox, det_bbox)
        
        # Convert IoU to distance (higher IoU = lower distance)
        return 1.0 - iou
    
    def _centroid_distance(self, tracked_object, detection):
        """Calculate centroid distance between tracked object and detection."""
        return self._euclidean_distance(tracked_object, detection)
    
    def _mask_iou_distance(self, tracked_object, detection):
        """Calculate mask IoU-based distance between tracked object and detection."""
        if tracked_object.last_detection is None:
            return float('inf')
        
        # Get masks
        track_mask = tracked_object.last_detection.data.get('mask')
        det_mask = detection.data.get('mask')
        
        if track_mask is None or det_mask is None:
            # Fall back to bbox IoU
            return self._iou_distance(tracked_object, detection)
        
        # Calculate mask IoU
        iou = self._calculate_mask_iou(track_mask, det_mask)
        
        # Convert IoU to distance
        return 1.0 - iou
    
    def _calculate_bbox_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_mask_iou(self, mask1, mask2):
        """Calculate Intersection over Union (IoU) between two masks."""
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        
        intersection_area = np.sum(intersection)
        union_area = np.sum(union)
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _mask_to_bbox(self, mask):
        """Convert mask to bounding box."""
        if not np.any(mask):
            return [0, 0, 0, 0]
        
        y_coords, x_coords = np.where(mask)
        x1, x2 = np.min(x_coords), np.max(x_coords)
        y1, y2 = np.min(y_coords), np.max(y_coords)
        
        return [x1, y1, x2, y2]
    
    def _get_mask_centroid(self, mask):
        """Get centroid of mask."""
        if not np.any(mask):
            return [0, 0]
        
        y_coords, x_coords = np.where(mask)
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        
        return [centroid_x, centroid_y]

    def get_tracker_info(self) -> Dict:
        """
        Get information about the tracker configuration.
        
        Returns:
            Dictionary with tracker information
        """
        return {
            'distance_function': self.distance_function_name,
            'distance_threshold': self.distance_threshold,
            'detection_threshold': self.detection_threshold,
            'hit_counter_max': self.tracker.hit_counter_max,
            'initialization_delay': self.tracker.initialization_delay,
            'active_tracks': self.get_track_count(),
            'frame_count': self.frame_count
        }
