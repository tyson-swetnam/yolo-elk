"""
Track Management Utilities

This module provides utilities for managing and analyzing tracking results,
including track statistics, filtering, and export functionality.
"""

import json
import csv
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import pandas as pd


class TrackManager:
    """
    Track management and analysis utilities.
    
    This class provides functionality for managing tracking results,
    calculating statistics, and exporting data.
    """
    
    def __init__(self):
        """Initialize the track manager."""
        self.track_history = {}
        self.frame_data = []
        self.statistics = {}
        
    def add_frame_data(self, frame_number: int, tracks: List[Dict], detections: Optional[List] = None):
        """
        Add tracking data for a frame.
        
        Args:
            frame_number: Frame number
            tracks: List of track dictionaries
            detections: Optional list of detections
        """
        frame_info = {
            'frame': frame_number,
            'tracks': tracks.copy() if tracks else [],
            'detections': detections.copy() if detections else [],
            'track_count': len([t for t in tracks if t.get('is_confirmed', False)]),
            'detection_count': len(detections) if detections else 0
        }
        
        self.frame_data.append(frame_info)
        
        # Update track history
        for track in tracks:
            track_id = track['track_id']
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'frame': frame_number,
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'class_id': track['class_id'],
                'center': track.get('center', [0, 0]),
                'velocity': track.get('velocity', [0, 0])
            })
    
    def get_track_statistics(self) -> Dict:
        """
        Calculate tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        if not self.frame_data:
            return {}
        
        total_frames = len(self.frame_data)
        total_tracks = len(self.track_history)
        
        # Track duration statistics
        track_durations = []
        track_lengths = []
        
        for track_id, history in self.track_history.items():
            duration = len(history)
            track_durations.append(duration)
            
            # Calculate track length (total distance traveled)
            if len(history) > 1:
                total_distance = 0
                for i in range(1, len(history)):
                    prev_center = history[i-1]['center']
                    curr_center = history[i]['center']
                    distance = np.sqrt(
                        (curr_center[0] - prev_center[0])**2 + 
                        (curr_center[1] - prev_center[1])**2
                    )
                    total_distance += distance
                track_lengths.append(total_distance)
        
        # Frame-level statistics
        tracks_per_frame = [frame['track_count'] for frame in self.frame_data]
        detections_per_frame = [frame['detection_count'] for frame in self.frame_data]
        
        statistics = {
            'total_frames': total_frames,
            'total_tracks': total_tracks,
            'avg_tracks_per_frame': np.mean(tracks_per_frame) if tracks_per_frame else 0,
            'max_tracks_per_frame': max(tracks_per_frame) if tracks_per_frame else 0,
            'min_tracks_per_frame': min(tracks_per_frame) if tracks_per_frame else 0,
            'avg_detections_per_frame': np.mean(detections_per_frame) if detections_per_frame else 0,
            'avg_track_duration': np.mean(track_durations) if track_durations else 0,
            'max_track_duration': max(track_durations) if track_durations else 0,
            'min_track_duration': min(track_durations) if track_durations else 0,
            'avg_track_length': np.mean(track_lengths) if track_lengths else 0,
            'max_track_length': max(track_lengths) if track_lengths else 0
        }
        
        self.statistics = statistics
        return statistics
    
    def filter_tracks_by_duration(self, min_duration: int = 1) -> Dict:
        """
        Filter tracks by minimum duration.
        
        Args:
            min_duration: Minimum number of frames a track must exist
            
        Returns:
            Filtered track history
        """
        filtered_tracks = {}
        
        for track_id, history in self.track_history.items():
            if len(history) >= min_duration:
                filtered_tracks[track_id] = history
        
        return filtered_tracks
    
    def filter_tracks_by_class(self, target_classes: List[int]) -> Dict:
        """
        Filter tracks by class ID.
        
        Args:
            target_classes: List of class IDs to keep
            
        Returns:
            Filtered track history
        """
        filtered_tracks = {}
        
        for track_id, history in self.track_history.items():
            if history and history[0]['class_id'] in target_classes:
                filtered_tracks[track_id] = history
        
        return filtered_tracks
    
    def get_track_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Get track trajectories as lists of center points.
        
        Returns:
            Dictionary mapping track IDs to lists of (x, y) coordinates
        """
        trajectories = {}
        
        for track_id, history in self.track_history.items():
            trajectory = []
            for entry in history:
                center = entry['center']
                trajectory.append((center[0], center[1]))
            trajectories[track_id] = trajectory
        
        return trajectories
    
    def export_to_json(self, filepath: str, include_frame_data: bool = True):
        """
        Export tracking data to JSON file.
        
        Args:
            filepath: Output file path
            include_frame_data: Whether to include frame-by-frame data
        """
        export_data = {
            'statistics': self.get_track_statistics(),
            'track_history': self.track_history
        }
        
        if include_frame_data:
            export_data['frame_data'] = self.frame_data
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)
    
    def export_to_csv(self, filepath: str, format_type: str = 'tracks'):
        """
        Export tracking data to CSV file.
        
        Args:
            filepath: Output file path
            format_type: 'tracks' for track-based format, 'frames' for frame-based format
        """
        if format_type == 'tracks':
            self._export_tracks_csv(filepath)
        elif format_type == 'frames':
            self._export_frames_csv(filepath)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
    
    def _export_tracks_csv(self, filepath: str):
        """Export track history to CSV."""
        rows = []
        
        for track_id, history in self.track_history.items():
            for entry in history:
                bbox = entry['bbox']
                center = entry['center']
                velocity = entry['velocity']
                
                rows.append({
                    'track_id': track_id,
                    'frame': entry['frame'],
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'center_x': center[0],
                    'center_y': center[1],
                    'velocity_x': velocity[0],
                    'velocity_y': velocity[1],
                    'confidence': entry['confidence'],
                    'class_id': entry['class_id']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def _export_frames_csv(self, filepath: str):
        """Export frame-based data to CSV."""
        rows = []
        
        for frame_info in self.frame_data:
            frame_num = frame_info['frame']
            
            for track in frame_info['tracks']:
                bbox = track['bbox']
                center = track.get('center', [0, 0])
                velocity = track.get('velocity', [0, 0])
                
                rows.append({
                    'frame': frame_num,
                    'track_id': track['track_id'],
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'center_x': center[0],
                    'center_y': center[1],
                    'velocity_x': velocity[0],
                    'velocity_y': velocity[1],
                    'confidence': track['confidence'],
                    'class_id': track['class_id'],
                    'is_confirmed': track.get('is_confirmed', False)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def export_trajectories(self, filepath: str, format_type: str = 'json'):
        """
        Export track trajectories.
        
        Args:
            filepath: Output file path
            format_type: 'json' or 'csv'
        """
        trajectories = self.get_track_trajectories()
        
        if format_type == 'json':
            with open(filepath, 'w') as f:
                json.dump(trajectories, f, indent=2)
        elif format_type == 'csv':
            rows = []
            for track_id, trajectory in trajectories.items():
                for i, (x, y) in enumerate(trajectory):
                    rows.append({
                        'track_id': track_id,
                        'point_index': i,
                        'x': x,
                        'y': y
                    })
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
    
    def analyze_track_patterns(self) -> Dict:
        """
        Analyze track movement patterns.
        
        Returns:
            Dictionary with pattern analysis results
        """
        patterns = {
            'stationary_tracks': [],
            'fast_moving_tracks': [],
            'direction_changes': {},
            'speed_statistics': {}
        }
        
        for track_id, history in self.track_history.items():
            if len(history) < 2:
                continue
            
            # Calculate speeds and direction changes
            speeds = []
            direction_changes = 0
            prev_direction = None
            
            for i in range(1, len(history)):
                prev_center = history[i-1]['center']
                curr_center = history[i]['center']
                
                # Calculate speed (distance per frame)
                distance = np.sqrt(
                    (curr_center[0] - prev_center[0])**2 + 
                    (curr_center[1] - prev_center[1])**2
                )
                speeds.append(distance)
                
                # Calculate direction
                if distance > 0:
                    direction = np.arctan2(
                        curr_center[1] - prev_center[1],
                        curr_center[0] - prev_center[0]
                    )
                    
                    if prev_direction is not None:
                        angle_diff = abs(direction - prev_direction)
                        if angle_diff > np.pi:
                            angle_diff = 2 * np.pi - angle_diff
                        
                        # Consider significant direction change if > 45 degrees
                        if angle_diff > np.pi / 4:
                            direction_changes += 1
                    
                    prev_direction = direction
            
            avg_speed = np.mean(speeds) if speeds else 0
            max_speed = max(speeds) if speeds else 0
            
            # Classify tracks
            if avg_speed < 1.0:  # Threshold for stationary
                patterns['stationary_tracks'].append(track_id)
            elif avg_speed > 10.0:  # Threshold for fast moving
                patterns['fast_moving_tracks'].append(track_id)
            
            patterns['direction_changes'][track_id] = direction_changes
            patterns['speed_statistics'][track_id] = {
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'total_distance': sum(speeds)
            }
        
        return patterns
    
    def get_class_distribution(self) -> Dict[int, int]:
        """
        Get distribution of track classes.
        
        Returns:
            Dictionary mapping class IDs to track counts
        """
        class_counts = {}
        
        for track_id, history in self.track_history.items():
            if history:
                class_id = history[0]['class_id']
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        return class_counts
    
    def reset(self):
        """Reset all tracking data."""
        self.track_history = {}
        self.frame_data = []
        self.statistics = {}
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def summary_report(self) -> str:
        """
        Generate a summary report of tracking results.
        
        Returns:
            Formatted summary string
        """
        stats = self.get_track_statistics()
        class_dist = self.get_class_distribution()
        patterns = self.analyze_track_patterns()
        
        report = []
        report.append("=" * 50)
        report.append("TRACKING SUMMARY REPORT")
        report.append("=" * 50)
        
        report.append(f"Total Frames: {stats.get('total_frames', 0)}")
        report.append(f"Total Tracks: {stats.get('total_tracks', 0)}")
        report.append(f"Average Tracks per Frame: {stats.get('avg_tracks_per_frame', 0):.2f}")
        report.append(f"Maximum Tracks per Frame: {stats.get('max_tracks_per_frame', 0)}")
        
        report.append("\nTrack Duration Statistics:")
        report.append(f"  Average Duration: {stats.get('avg_track_duration', 0):.2f} frames")
        report.append(f"  Maximum Duration: {stats.get('max_track_duration', 0)} frames")
        report.append(f"  Minimum Duration: {stats.get('min_track_duration', 0)} frames")
        
        report.append("\nClass Distribution:")
        for class_id, count in class_dist.items():
            report.append(f"  Class {class_id}: {count} tracks")
        
        report.append("\nMovement Patterns:")
        report.append(f"  Stationary Tracks: {len(patterns['stationary_tracks'])}")
        report.append(f"  Fast Moving Tracks: {len(patterns['fast_moving_tracks'])}")
        
        report.append("=" * 50)
        
        return "\n".join(report)
