"""
Norfair-based Object Tracker

This module provides a wrapper around the Norfair tracking library
for multi-object tracking with YOLO detections.
"""

import numpy as np
from typing import List, Dict, Optional, Callable
import norfair
from norfair import Detection, Tracker, draw_tracked_objects


class NorfairTracker:
    """
    Norfair-based multi-object tracker.
    
    This class provides an interface to Norfair tracking with customizable
    distance functions and tracking parameters.
    """
    
    def __init__(
        self,
        distance_function: str = "euclidean",
        distance_threshold: float = 30,
        hit_counter_max: int = 15,
        initialization_delay: int = 3,
        pointwise_hit_counter_max: int = 4,
        detection_threshold: float = 0.1,
        past_detections_length: int = 4,
        reid_distance_function: Optional[Callable] = None,
        reid_distance_threshold: float = 0.3
    ):
        """
        Initialize the Norfair tracker.
        
        Args:
            distance_function: Distance function type ('euclidean', 'iou', 'centroid')
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
        detections: np.ndarray,
        frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, score, class_id]
            frame: Optional frame for visualization
            
        Returns:
            List of tracked objects with track information
        """
        self.frame_count += 1
        
        # Convert detections to Norfair format
        norfair_detections = self._convert_detections(detections)
        
        # Update tracker
        tracked_objects = self.tracker.update(
            detections=norfair_detections,
            period=1
        )
        
        # Convert back to our format
        tracks = self._convert_tracked_objects(tracked_objects)
        
        return tracks
    
    def _convert_detections(self, detections: np.ndarray) -> List[Detection]:
        """
        Convert detection array to Norfair Detection objects.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, score, class_id]
            
        Returns:
            List of Norfair Detection objects
        """
        norfair_detections = []
        
        for det in detections:
            if len(det) >= 5 and det[4] >= self.detection_threshold:
                x1, y1, x2, y2, score = det[:5]
                class_id = int(det[5]) if len(det) > 5 else 0
                
                # Convert to center point and dimensions
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                # Create detection points (center + corners for better tracking)
                points = np.array([
                    [center_x, center_y],  # Center
                    [x1, y1],              # Top-left
                    [x2, y2]               # Bottom-right
                ])
                
                detection = Detection(
                    points=points,
                    scores=np.array([score, score, score]),
                    data={
                        'bbox': [x1, y1, x2, y2],
                        'confidence': score,
                        'class_id': class_id,
                        'width': width,
                        'height': height
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
                    'bbox': detection_data.get('bbox', [0, 0, 0, 0]),
                    'confidence': detection_data.get('confidence', 0.0),
                    'class_id': detection_data.get('class_id', 0),
                    'age': obj.age,
                    'hit_counter': obj.hit_counter,
                    'last_distance': obj.last_distance,
                    'is_confirmed': obj.age >= self.tracker.initialization_delay,
                    'center': obj.estimate.tolist() if obj.estimate is not None else [0, 0],
                    'velocity': obj.velocity.tolist() if obj.velocity is not None else [0, 0],
                    'frame_count': self.frame_count
                }
                
                tracks.append(track_info)
        
        return tracks
    
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        draw_labels: bool = True,
        draw_ids: bool = True,
        draw_trails: bool = False
    ) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            tracks: List of track dictionaries
            draw_labels: Whether to draw class labels
            draw_ids: Whether to draw track IDs
            draw_trails: Whether to draw tracking trails
            
        Returns:
            Frame with tracking visualization
        """
        # Convert tracks back to Norfair format for drawing
        tracked_objects = []
        
        for track in tracks:
            # Create a mock TrackedObject for drawing
            class MockTrackedObject:
                def __init__(self, track_info):
                    self.id = track_info['track_id']
                    self.estimate = np.array(track_info['center'])
                    self.last_detection = type('obj', (object,), {
                        'data': {
                            'bbox': track_info['bbox'],
                            'confidence': track_info['confidence'],
                            'class_id': track_info['class_id']
                        }
                    })()
                    
            tracked_objects.append(MockTrackedObject(track))
        
        # Use Norfair's drawing function
        if tracked_objects:
            frame = draw_tracked_objects(
                frame,
                tracked_objects,
                draw_labels=draw_labels,
                draw_ids=draw_ids
            )
        
        return frame
    
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
        det_center = detection.points[0]  # First point is center
        
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
        iou = self._calculate_iou(track_bbox, det_bbox)
        
        # Convert IoU to distance (higher IoU = lower distance)
        return 1.0 - iou
    
    def _centroid_distance(self, tracked_object, detection):
        """Calculate centroid distance between tracked object and detection."""
        return self._euclidean_distance(tracked_object, detection)
    
    def _calculate_iou(self, bbox1, bbox2):
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
