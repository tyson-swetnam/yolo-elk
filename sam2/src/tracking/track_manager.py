"""
Track Management Utilities for SAM2

This module provides utilities for managing and analyzing SAM2 mask-based tracking results,
including track statistics, filtering, and export functionality.
"""

import json
import csv
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import cv2


class TrackManager:
    """
    Track management and analysis utilities for SAM2 segmentation tracking.
    
    This class provides functionality for managing mask-based tracking results,
    calculating statistics, and exporting data.
    """
    
    def __init__(self):
        """Initialize the track manager."""
        self.track_history = {}
        self.frame_data = []
        self.statistics = {}
        
    def add_frame_data(self, frame_number: int, tracks: List[Dict], segmentations: Optional[List] = None):
        """
        Add tracking data for a frame.
        
        Args:
            frame_number: Frame number
            tracks: List of track dictionaries
            segmentations: Optional list of segmentations
        """
        frame_info = {
            'frame': frame_number,
            'tracks': tracks.copy() if tracks else [],
            'segmentations': segmentations.copy() if segmentations else [],
            'track_count': len([t for t in tracks if t.get('is_confirmed', False)]),
            'segmentation_count': len(segmentations) if segmentations else 0,
            'elk_types': self._count_elk_types(tracks)
        }
        
        self.frame_data.append(frame_info)
        
        # Update track history
        for track in tracks:
            track_id = track['track_id']
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            # Store mask as compressed format to save memory
            mask = track.get('mask')
            mask_data = None
            if mask is not None:
                # Store mask as run-length encoding or compressed format
                mask_data = self._compress_mask(mask)
            
            self.track_history[track_id].append({
                'frame': frame_number,
                'bbox': track['bbox'],
                'confidence': track['confidence'],
                'elk_confidence': track.get('elk_confidence', track['confidence']),
                'elk_type': track.get('elk_type', 'unknown'),
                'features': track.get('features', {}),
                'area': track.get('area', 0),
                'centroid': track.get('centroid', [0, 0]),
                'velocity': track.get('velocity', [0, 0]),
                'mask_data': mask_data
            })
    
    def _count_elk_types(self, tracks: List[Dict]) -> Dict[str, int]:
        """Count elk types in tracks."""
        elk_counts = {'bulls': 0, 'cows': 0, 'calves': 0, 'unknown': 0}
        
        for track in tracks:
            if track.get('is_confirmed', False):
                elk_type = track.get('elk_type', 'unknown')
                if elk_type == 'bull':
                    elk_counts['bulls'] += 1
                elif elk_type == 'cow':
                    elk_counts['cows'] += 1
                elif elk_type == 'calf':
                    elk_counts['calves'] += 1
                else:
                    elk_counts['unknown'] += 1
        
        return elk_counts
    
    def _compress_mask(self, mask: np.ndarray) -> Dict:
        """Compress mask for storage."""
        # Simple run-length encoding
        if mask is None or not np.any(mask):
            return None
        
        # Get bounding box to reduce storage
        y_coords, x_coords = np.where(mask)
        if len(y_coords) == 0:
            return None
        
        x1, x2 = np.min(x_coords), np.max(x_coords)
        y1, y2 = np.min(y_coords), np.max(y_coords)
        
        # Store only the bounding box region
        mask_roi = mask[y1:y2+1, x1:x2+1]
        
        return {
            'bbox': [x1, y1, x2, y2],
            'shape': mask.shape,
            'roi_shape': mask_roi.shape,
            'roi_data': mask_roi.astype(np.uint8).tobytes()
        }
    
    def _decompress_mask(self, mask_data: Dict) -> np.ndarray:
        """Decompress mask from storage."""
        if mask_data is None:
            return None
        
        # Reconstruct mask
        full_shape = mask_data['shape']
        roi_shape = mask_data['roi_shape']
        bbox = mask_data['bbox']
        
        # Create full mask
        full_mask = np.zeros(full_shape, dtype=bool)
        
        # Reconstruct ROI
        roi_data = np.frombuffer(mask_data['roi_data'], dtype=np.uint8)
        roi_mask = roi_data.reshape(roi_shape).astype(bool)
        
        # Place ROI in full mask
        x1, y1, x2, y2 = bbox
        full_mask[y1:y2+1, x1:x2+1] = roi_mask
        
        return full_mask
    
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
        track_areas = []
        
        for track_id, history in self.track_history.items():
            duration = len(history)
            track_durations.append(duration)
            
            # Calculate track length (total distance traveled)
            if len(history) > 1:
                total_distance = 0
                areas = []
                for i in range(1, len(history)):
                    prev_centroid = history[i-1]['centroid']
                    curr_centroid = history[i]['centroid']
                    distance = np.sqrt(
                        (curr_centroid[0] - prev_centroid[0])**2 + 
                        (curr_centroid[1] - prev_centroid[1])**2
                    )
                    total_distance += distance
                    areas.append(history[i]['area'])
                
                track_lengths.append(total_distance)
                if areas:
                    track_areas.extend(areas)
        
        # Frame-level statistics
        tracks_per_frame = [frame['track_count'] for frame in self.frame_data]
        segmentations_per_frame = [frame['segmentation_count'] for frame in self.frame_data]
        
        # Elk type statistics
        elk_type_stats = {'bulls': [], 'cows': [], 'calves': [], 'unknown': []}
        for frame in self.frame_data:
            elk_types = frame['elk_types']
            for elk_type, count in elk_types.items():
                elk_type_stats[elk_type].append(count)
        
        statistics = {
            'total_frames': total_frames,
            'total_tracks': total_tracks,
            'avg_tracks_per_frame': np.mean(tracks_per_frame) if tracks_per_frame else 0,
            'max_tracks_per_frame': max(tracks_per_frame) if tracks_per_frame else 0,
            'min_tracks_per_frame': min(tracks_per_frame) if tracks_per_frame else 0,
            'avg_segmentations_per_frame': np.mean(segmentations_per_frame) if segmentations_per_frame else 0,
            'avg_track_duration': np.mean(track_durations) if track_durations else 0,
            'max_track_duration': max(track_durations) if track_durations else 0,
            'min_track_duration': min(track_durations) if track_durations else 0,
            'avg_track_length': np.mean(track_lengths) if track_lengths else 0,
            'max_track_length': max(track_lengths) if track_lengths else 0,
            'avg_track_area': np.mean(track_areas) if track_areas else 0,
            'max_track_area': max(track_areas) if track_areas else 0,
            'elk_type_statistics': {
                elk_type: {
                    'avg_per_frame': np.mean(counts) if counts else 0,
                    'max_per_frame': max(counts) if counts else 0,
                    'total_tracks': len([t for t in self.track_history.values() 
                                       if t and t[0].get('elk_type') == elk_type])
                }
                for elk_type, counts in elk_type_stats.items()
            }
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
    
    def filter_tracks_by_elk_type(self, target_types: List[str]) -> Dict:
        """
        Filter tracks by elk type.
        
        Args:
            target_types: List of elk types to keep ('bull', 'cow', 'calf', 'unknown')
            
        Returns:
            Filtered track history
        """
        filtered_tracks = {}
        
        for track_id, history in self.track_history.items():
            if history and history[0]['elk_type'] in target_types:
                filtered_tracks[track_id] = history
        
        return filtered_tracks
    
    def filter_tracks_by_confidence(self, min_confidence: float = 0.5) -> Dict:
        """
        Filter tracks by minimum elk confidence.
        
        Args:
            min_confidence: Minimum elk confidence threshold
            
        Returns:
            Filtered track history
        """
        filtered_tracks = {}
        
        for track_id, history in self.track_history.items():
            # Check if track has sufficient confidence
            avg_confidence = np.mean([entry['elk_confidence'] for entry in history])
            if avg_confidence >= min_confidence:
                filtered_tracks[track_id] = history
        
        return filtered_tracks
    
    def get_track_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """
        Get track trajectories as lists of centroid points.
        
        Returns:
            Dictionary mapping track IDs to lists of (x, y) coordinates
        """
        trajectories = {}
        
        for track_id, history in self.track_history.items():
            trajectory = []
            for entry in history:
                centroid = entry['centroid']
                trajectory.append((centroid[0], centroid[1]))
            trajectories[track_id] = trajectory
        
        return trajectories
    
    def get_track_masks(self, track_id: int) -> List[np.ndarray]:
        """
        Get all masks for a specific track.
        
        Args:
            track_id: Track ID
            
        Returns:
            List of masks for the track
        """
        if track_id not in self.track_history:
            return []
        
        masks = []
        for entry in self.track_history[track_id]:
            mask_data = entry.get('mask_data')
            if mask_data:
                mask = self._decompress_mask(mask_data)
                masks.append(mask)
            else:
                masks.append(None)
        
        return masks
    
    def export_to_json(self, filepath: str, include_frame_data: bool = True, include_masks: bool = False):
        """
        Export tracking data to JSON file.
        
        Args:
            filepath: Output file path
            include_frame_data: Whether to include frame-by-frame data
            include_masks: Whether to include mask data (increases file size significantly)
        """
        export_data = {
            'statistics': self.get_track_statistics(),
            'track_history': self._prepare_track_history_for_export(include_masks)
        }
        
        if include_frame_data:
            export_data['frame_data'] = self.frame_data
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=self._json_serializer)
    
    def _prepare_track_history_for_export(self, include_masks: bool = False) -> Dict:
        """Prepare track history for JSON export."""
        export_history = {}
        
        for track_id, history in self.track_history.items():
            export_history[track_id] = []
            for entry in history:
                export_entry = entry.copy()
                
                if not include_masks and 'mask_data' in export_entry:
                    del export_entry['mask_data']
                
                export_history[track_id].append(export_entry)
        
        return export_history
    
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
                centroid = entry['centroid']
                velocity = entry['velocity']
                features = entry.get('features', {})
                
                rows.append({
                    'track_id': track_id,
                    'frame': entry['frame'],
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'velocity_x': velocity[0],
                    'velocity_y': velocity[1],
                    'confidence': entry['confidence'],
                    'elk_confidence': entry['elk_confidence'],
                    'elk_type': entry['elk_type'],
                    'area': entry['area'],
                    'has_elk_colors': features.get('has_elk_colors', False),
                    'elk_color_ratio': features.get('elk_color_ratio', 0),
                    'aspect_ratio': features.get('aspect_ratio', 0),
                    'white_rump_score': features.get('white_rump_score', 0),
                    'compactness': features.get('compactness', 0)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def _export_frames_csv(self, filepath: str):
        """Export frame-based data to CSV."""
        rows = []
        
        for frame_info in self.frame_data:
            frame_num = frame_info['frame']
            elk_types = frame_info['elk_types']
            
            for track in frame_info['tracks']:
                bbox = track['bbox']
                centroid = track.get('centroid', [0, 0])
                velocity = track.get('velocity', [0, 0])
                features = track.get('features', {})
                
                rows.append({
                    'frame': frame_num,
                    'track_id': track['track_id'],
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3],
                    'centroid_x': centroid[0],
                    'centroid_y': centroid[1],
                    'velocity_x': velocity[0],
                    'velocity_y': velocity[1],
                    'confidence': track['confidence'],
                    'elk_confidence': track.get('elk_confidence', track['confidence']),
                    'elk_type': track.get('elk_type', 'unknown'),
                    'area': track.get('area', 0),
                    'is_confirmed': track.get('is_confirmed', False),
                    'frame_bulls': elk_types['bulls'],
                    'frame_cows': elk_types['cows'],
                    'frame_calves': elk_types['calves'],
                    'frame_unknown': elk_types['unknown']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def analyze_elk_behavior(self) -> Dict:
        """
        Analyze elk behavior patterns from tracking data.
        
        Returns:
            Dictionary with behavior analysis results
        """
        behavior = {
            'stationary_elk': [],
            'fast_moving_elk': [],
            'direction_changes': {},
            'speed_statistics': {},
            'size_changes': {},
            'elk_interactions': []
        }
        
        for track_id, history in self.track_history.items():
            if len(history) < 2:
                continue
            
            # Calculate speeds and direction changes
            speeds = []
            areas = []
            direction_changes = 0
            prev_direction = None
            
            for i in range(1, len(history)):
                prev_centroid = history[i-1]['centroid']
                curr_centroid = history[i]['centroid']
                
                # Calculate speed (distance per frame)
                distance = np.sqrt(
                    (curr_centroid[0] - prev_centroid[0])**2 + 
                    (curr_centroid[1] - prev_centroid[1])**2
                )
                speeds.append(distance)
                areas.append(history[i]['area'])
                
                # Calculate direction
                if distance > 0:
                    direction = np.arctan2(
                        curr_centroid[1] - prev_centroid[1],
                        curr_centroid[0] - prev_centroid[0]
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
            area_variance = np.var(areas) if areas else 0
            
            # Classify elk behavior
            if avg_speed < 2.0:  # Threshold for stationary
                behavior['stationary_elk'].append(track_id)
            elif avg_speed > 15.0:  # Threshold for fast moving
                behavior['fast_moving_elk'].append(track_id)
            
            behavior['direction_changes'][track_id] = direction_changes
            behavior['speed_statistics'][track_id] = {
                'avg_speed': avg_speed,
                'max_speed': max_speed,
                'total_distance': sum(speeds)
            }
            behavior['size_changes'][track_id] = {
                'area_variance': area_variance,
                'avg_area': np.mean(areas) if areas else 0
            }
        
        return behavior
    
    def get_elk_type_distribution(self) -> Dict[str, int]:
        """
        Get distribution of elk types.
        
        Returns:
            Dictionary mapping elk types to track counts
        """
        type_counts = {}
        
        for track_id, history in self.track_history.items():
            if history:
                elk_type = history[0]['elk_type']
                type_counts[elk_type] = type_counts.get(elk_type, 0) + 1
        
        return type_counts
    
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
        elif isinstance(obj, bytes):
            return obj.hex()  # Convert bytes to hex string
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def summary_report(self) -> str:
        """
        Generate a summary report of elk tracking results.
        
        Returns:
            Formatted summary string
        """
        stats = self.get_track_statistics()
        type_dist = self.get_elk_type_distribution()
        behavior = self.analyze_elk_behavior()
        
        report = []
        report.append("=" * 60)
        report.append("SAM2 ELK TRACKING SUMMARY REPORT")
        report.append("=" * 60)
        
        report.append(f"Total Frames: {stats.get('total_frames', 0)}")
        report.append(f"Total Tracks: {stats.get('total_tracks', 0)}")
        report.append(f"Average Tracks per Frame: {stats.get('avg_tracks_per_frame', 0):.2f}")
        report.append(f"Maximum Tracks per Frame: {stats.get('max_tracks_per_frame', 0)}")
        
        report.append("\nTrack Duration Statistics:")
        report.append(f"  Average Duration: {stats.get('avg_track_duration', 0):.2f} frames")
        report.append(f"  Maximum Duration: {stats.get('max_track_duration', 0)} frames")
        report.append(f"  Minimum Duration: {stats.get('min_track_duration', 0)} frames")
        
        report.append("\nElk Type Distribution:")
        for elk_type, count in type_dist.items():
            report.append(f"  {elk_type.title()}: {count} tracks")
        
        report.append("\nElk Type Statistics (per frame):")
        elk_stats = stats.get('elk_type_statistics', {})
        for elk_type, type_stats in elk_stats.items():
            report.append(f"  {elk_type.title()}:")
            report.append(f"    Average per frame: {type_stats['avg_per_frame']:.2f}")
            report.append(f"    Maximum per frame: {type_stats['max_per_frame']}")
            report.append(f"    Total tracks: {type_stats['total_tracks']}")
        
        report.append("\nBehavior Analysis:")
        report.append(f"  Stationary Elk: {len(behavior['stationary_elk'])}")
        report.append(f"  Fast Moving Elk: {len(behavior['fast_moving_elk'])}")
        
        report.append("\nSegmentation Statistics:")
        report.append(f"  Average Area: {stats.get('avg_track_area', 0):.0f} pixels")
        report.append(f"  Maximum Area: {stats.get('max_track_area', 0):.0f} pixels")
        
        report.append("=" * 60)
        
        return "\n".join(report)
