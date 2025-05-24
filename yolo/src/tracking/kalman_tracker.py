"""
Kalman Filter-based Object Tracker

This module provides a Kalman filter implementation for object tracking
with motion prediction and state estimation.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


class KalmanTracker:
    """
    Kalman filter-based multi-object tracker.
    
    This class implements a simple multi-object tracker using Kalman filters
    for motion prediction and Hungarian algorithm for data association.
    """
    
    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 50,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """
        Initialize the Kalman tracker.
        
        Args:
            max_disappeared: Maximum frames a track can be missing before deletion
            max_distance: Maximum distance for track-detection association
            min_hits: Minimum hits before a track is considered confirmed
            iou_threshold: IoU threshold for association
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
        
    def _create_kalman_filter(self, bbox: List[float]) -> KalmanFilter:
        """
        Create a Kalman filter for tracking a bounding box.
        
        Args:
            bbox: Initial bounding box [x1, y1, x2, y2]
            
        Returns:
            Configured Kalman filter
        """
        # State vector: [x, y, w, h, vx, vy, vw, vh]
        # x, y: center coordinates
        # w, h: width and height
        # vx, vy, vw, vh: velocities
        
        kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        kf.Q *= 0.01
        
        # Measurement noise covariance
        kf.R *= 10
        
        # Initial state covariance
        kf.P *= 1000
        
        # Initialize state
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        kf.x = np.array([cx, cy, w, h, 0, 0, 0, 0])
        
        return kf
    
    def _bbox_to_measurement(self, bbox: List[float]) -> np.ndarray:
        """Convert bounding box to measurement vector."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return np.array([cx, cy, w, h])
    
    def _measurement_to_bbox(self, measurement: np.ndarray) -> List[float]:
        """Convert measurement vector to bounding box."""
        cx, cy, w, h = measurement
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
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
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Euclidean distance between bbox centers."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2
        
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    
    def update(self, detections: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: Array of detections [x1, y1, x2, y2, score, class_id]
            
        Returns:
            List of tracked objects with track information
        """
        self.frame_count += 1
        
        # Predict step for all tracks
        for track_id in self.tracks:
            self.tracks[track_id]['kf'].predict()
            self.tracks[track_id]['time_since_update'] += 1
        
        # Prepare detections
        det_bboxes = []
        det_scores = []
        det_classes = []
        
        for det in detections:
            if len(det) >= 5:
                det_bboxes.append(det[:4].tolist())
                det_scores.append(det[4])
                det_classes.append(int(det[5]) if len(det) > 5 else 0)
        
        # Data association
        if len(det_bboxes) > 0 and len(self.tracks) > 0:
            matched, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
                det_bboxes, list(self.tracks.keys())
            )
        else:
            matched = []
            unmatched_dets = list(range(len(det_bboxes)))
            unmatched_trks = list(self.tracks.keys())
        
        # Update matched tracks
        for det_idx, track_id in matched:
            measurement = self._bbox_to_measurement(det_bboxes[det_idx])
            self.tracks[track_id]['kf'].update(measurement)
            self.tracks[track_id]['time_since_update'] = 0
            self.tracks[track_id]['hit_streak'] += 1
            self.tracks[track_id]['confidence'] = det_scores[det_idx]
            self.tracks[track_id]['class_id'] = det_classes[det_idx]
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_new_track(det_bboxes[det_idx], det_scores[det_idx], det_classes[det_idx])
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id in unmatched_trks:
            if self.tracks[track_id]['time_since_update'] > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        # Prepare output
        return self._get_track_results()
    
    def _associate_detections_to_tracks(
        self, 
        detections: List[List[float]], 
        track_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to tracks using Hungarian algorithm.
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if len(track_ids) == 0:
            return [], list(range(len(detections))), []
        
        # Calculate cost matrix
        cost_matrix = np.zeros((len(detections), len(track_ids)))
        
        for d, det_bbox in enumerate(detections):
            for t, track_id in enumerate(track_ids):
                # Get predicted bbox from Kalman filter
                pred_state = self.tracks[track_id]['kf'].x
                pred_bbox = self._measurement_to_bbox(pred_state[:4])
                
                # Calculate IoU
                iou = self._calculate_iou(det_bbox, pred_bbox)
                
                # Use negative IoU as cost (higher IoU = lower cost)
                cost_matrix[d, t] = 1 - iou
        
        # Solve assignment problem
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        # Filter matches based on IoU threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(track_ids)))
        
        for d, t in zip(det_indices, track_indices):
            if cost_matrix[d, t] <= (1 - self.iou_threshold):
                matched.append((d, track_ids[t]))
                unmatched_dets.remove(d)
                unmatched_trks.remove(t)
        
        unmatched_trks = [track_ids[i] for i in unmatched_trks]
        
        return matched, unmatched_dets, unmatched_trks
    
    def _create_new_track(self, bbox: List[float], score: float, class_id: int):
        """Create a new track."""
        kf = self._create_kalman_filter(bbox)
        
        self.tracks[self.next_id] = {
            'kf': kf,
            'time_since_update': 0,
            'hit_streak': 1,
            'confidence': score,
            'class_id': class_id,
            'age': 1
        }
        
        self.next_id += 1
    
    def _get_track_results(self) -> List[Dict]:
        """Get current tracking results."""
        results = []
        
        for track_id, track in self.tracks.items():
            # Get current state
            state = track['kf'].x
            bbox = self._measurement_to_bbox(state[:4])
            
            # Only return confirmed tracks
            if track['hit_streak'] >= self.min_hits and track['time_since_update'] <= 1:
                results.append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': track['confidence'],
                    'class_id': track['class_id'],
                    'age': track['age'],
                    'hit_streak': track['hit_streak'],
                    'time_since_update': track['time_since_update'],
                    'is_confirmed': track['hit_streak'] >= self.min_hits,
                    'center': [state[0], state[1]],
                    'velocity': [state[4], state[5]],
                    'frame_count': self.frame_count
                })
            
            # Update age
            track['age'] += 1
        
        return results
    
    def get_track_count(self) -> int:
        """Get current number of active tracks."""
        return len([t for t in self.tracks.values() 
                   if t['hit_streak'] >= self.min_hits and t['time_since_update'] <= 1])
    
    def reset(self):
        """Reset the tracker."""
        self.tracks = {}
        self.next_id = 0
        self.frame_count = 0
    
    def get_tracker_info(self) -> Dict:
        """
        Get information about the tracker configuration.
        
        Returns:
            Dictionary with tracker information
        """
        return {
            'tracker_type': 'KalmanTracker',
            'max_disappeared': self.max_disappeared,
            'max_distance': self.max_distance,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold,
            'active_tracks': self.get_track_count(),
            'total_tracks': len(self.tracks),
            'frame_count': self.frame_count
        }
