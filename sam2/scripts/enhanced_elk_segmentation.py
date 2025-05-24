#!/usr/bin/env python3
"""
Enhanced Real-time Elk Segmentation Script using SAM2

This script opens video files and uses the enhanced ElkSegmenter
to identify and classify elk (bulls, cows, calves) in real-time with motion
compensation for helicopter footage using SAM2 segmentation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np
import time
import yaml
from src.segmentation.elk_segmenter import ElkSegmenter
from src.tracking.norfair_tracker import NorfairTracker
from src.tracking.track_manager import TrackManager


class EnhancedElkSegmentationPlayer:
    """Enhanced real-time elk segmentation with helicopter motion compensation using SAM2."""
    
    def __init__(self, video_path, config_path=None, confidence_threshold=0.5):
        """
        Initialize the enhanced elk segmenter.
        
        Args:
            video_path: Path to the video file
            config_path: Path to configuration file
            confidence_threshold: Minimum confidence for segmentations
        """
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize enhanced elk segmenter
        print("Initializing Enhanced SAM2 Elk Segmenter...")
        self.segmenter = ElkSegmenter(
            model_checkpoint=self.config['segmentation']['model_checkpoint'],
            model_config=self.config['segmentation']['model_config'],
            device=self.config['segmentation']['device'],
            confidence_threshold=confidence_threshold,
            mask_threshold=self.config['segmentation']['mask_threshold']
        )
        
        # Initialize tracker
        print("Initializing Tracker...")
        tracker_config = self.config['tracking']['norfair']
        self.tracker = NorfairTracker(
            distance_function=tracker_config['distance_function'],
            distance_threshold=tracker_config['distance_threshold'],
            hit_counter_max=tracker_config['hit_counter_max'],
            initialization_delay=tracker_config['initialization_delay'],
            detection_threshold=tracker_config['detection_threshold']
        )
        
        # Initialize track manager
        self.track_manager = TrackManager()
        
        # Tracking and statistics
        self.elk_history = []
        self.max_elk_counts = {'bulls': 0, 'cows': 0, 'calves': 0, 'total': 0}
        
        # Display settings
        self.show_motion_compensation = self.config['motion_compensation']['enable']
        self.show_feature_analysis = self.config['feature_analysis']['enable']
        self.show_masks = self.config['visualization']['show_masks']
        self.show_tracking = self.config['tracking']['enable_tracking']
        self.mask_alpha = self.config['visualization']['mask_alpha']
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = project_root / "configs" / "default_config.yaml"
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def draw_elk_segmentation(self, frame: np.ndarray, elk_segmentation: dict) -> np.ndarray:
        """
        Draw a single elk segmentation with enhanced visualization.
        
        Args:
            frame: Input frame
            elk_segmentation: Elk segmentation dictionary
            
        Returns:
            Frame with drawn segmentation
        """
        mask = elk_segmentation['mask']
        elk_type = elk_segmentation['elk_type']
        confidence = elk_segmentation['elk_confidence']
        features = elk_segmentation['features']
        centroid = elk_segmentation['centroid']
        
        # Get color for elk type
        color = self.segmenter.get_elk_color(elk_type)
        
        # Draw mask if enabled
        if self.show_masks and mask is not None:
            mask_colored = np.zeros_like(frame)
            mask_colored[mask] = color
            frame = cv2.addWeighted(frame, 1 - self.mask_alpha, mask_colored, self.mask_alpha, 0)
        
        # Draw bounding box
        bbox = elk_segmentation.get('bbox', self.segmenter._mask_to_bbox(mask))
        if bbox != [0, 0, 0, 0]:
            x1, y1, x2, y2 = map(int, bbox)
            thickness = max(2, int(confidence * 5))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare detailed label
        label_lines = [
            f"{elk_type.upper()}: {confidence:.2f}"
        ]
        
        # Add feature information if available
        if self.show_feature_analysis and features:
            if features.get('has_elk_colors', False):
                label_lines.append("✓ Elk Colors")
            if features.get('white_rump_score', 0) > 0.1:
                label_lines.append("✓ White Rump")
            if features.get('compactness', 0) > 0.3:
                label_lines.append("✓ Good Shape")
        
        # Draw labels at centroid
        if centroid != [0, 0]:
            label_pos = (int(centroid[0]), int(centroid[1]))
            
            # Calculate label background size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            
            max_width = 0
            total_height = 0
            line_heights = []
            
            for line in label_lines:
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
                max_width = max(max_width, text_width)
                line_heights.append(text_height + baseline)
                total_height += text_height + baseline + 5
            
            # Draw label background
            label_y = label_pos[1] - total_height - 10
            if label_y < 0:
                label_y = label_pos[1] + 10
            
            cv2.rectangle(
                frame,
                (label_pos[0] - 5, label_y),
                (label_pos[0] + max_width + 10, label_y + total_height + 10),
                color,
                -1
            )
            
            # Draw label text
            current_y = label_y + 20
            for i, line in enumerate(label_lines):
                cv2.putText(
                    frame,
                    line,
                    (label_pos[0], current_y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness
                )
                current_y += line_heights[i] + 5
        
        return frame
    
    def draw_elk_segmentations(self, frame: np.ndarray, elk_results: dict) -> np.ndarray:
        """
        Draw all elk segmentations on the frame.
        
        Args:
            frame: Input frame
            elk_results: Results from elk segmenter
            
        Returns:
            Frame with all segmentations drawn
        """
        annotated_frame = frame.copy()
        
        # Draw each elk segmentation
        for elk_segmentation in elk_results['elk_segmentations']:
            annotated_frame = self.draw_elk_segmentation(annotated_frame, elk_segmentation)
        
        return annotated_frame
    
    def draw_tracking_results(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """
        Draw tracking results on frame.
        
        Args:
            frame: Input frame
            tracks: List of track dictionaries
            
        Returns:
            Frame with tracking visualization
        """
        if not self.show_tracking or not tracks:
            return frame
        
        return self.tracker.draw_tracks(
            frame, 
            tracks,
            draw_masks=self.show_masks,
            draw_labels=True,
            draw_ids=True,
            mask_alpha=self.mask_alpha
        )
    
    def create_info_panel(self, frame: np.ndarray, elk_results: dict, tracks: list,
                         frame_num: int, total_frames: int, fps: float) -> np.ndarray:
        """
        Create comprehensive information panel.
        
        Args:
            frame: Input frame
            elk_results: Results from elk segmenter
            tracks: Tracking results
            frame_num: Current frame number
            total_frames: Total frames in video
            fps: Processing FPS
            
        Returns:
            Frame with info panel
        """
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Main info panel
        panel_width = 350
        panel_height = 220
        
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        
        # Elk count panel
        count_panel_y = panel_height + 20
        count_panel_height = 140
        cv2.rectangle(overlay, (10, count_panel_y), (panel_width, count_panel_y + count_panel_height), (0, 0, 0), -1)
        
        # Motion/quality panel
        quality_panel_x = panel_width + 20
        quality_panel_width = 280
        quality_panel_height = 120
        cv2.rectangle(overlay, (quality_panel_x, 10), (quality_panel_x + quality_panel_width, quality_panel_height), (0, 0, 0), -1)
        
        # Tracking panel
        tracking_panel_y = quality_panel_height + 20
        tracking_panel_height = 100
        cv2.rectangle(overlay, (quality_panel_x, tracking_panel_y), (quality_panel_x + quality_panel_width, tracking_panel_y + tracking_panel_height), (0, 0, 0), -1)
        
        # Blend with original frame
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Main info text
        info_lines = [
            f"Frame: {frame_num}/{total_frames}",
            f"Progress: {frame_num/total_frames*100:.1f}%",
            f"Processing FPS: {fps:.1f}",
            f"Total Elk: {elk_results['total_elk_count']}",
            f"Active Tracks: {len([t for t in tracks if t.get('is_confirmed', False)])}",
            "",
            "Controls:",
            "SPACE: Pause/Resume",
            "M: Toggle Motion Compensation",
            "F: Toggle Feature Analysis", 
            "S: Toggle Masks",
            "T: Toggle Tracking",
            "Q/ESC: Quit"
        ]
        
        y_offset = 25
        for line in info_lines:
            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 15
        
        # Elk count breakdown
        elk_types = elk_results['elk_types']
        count_lines = [
            "Current Frame:",
            f"Bulls: {elk_types['bulls']}",
            f"Cows: {elk_types['cows']}",
            f"Calves: {elk_types['calves']}",
            "",
            "Maximum Seen:",
            f"Bulls: {self.max_elk_counts['bulls']}",
            f"Cows: {self.max_elk_counts['cows']}",
            f"Calves: {self.max_elk_counts['calves']}",
            "",
            "Tracking Stats:",
            f"Total Tracks: {self.track_manager.get_track_statistics().get('total_tracks', 0)}"
        ]
        
        y_offset = count_panel_y + 20
        for line in count_lines:
            color = (255, 255, 255)
            if "Bulls:" in line and elk_types['bulls'] > 0:
                color = (0, 255, 0)  # Green
            elif "Cows:" in line and elk_types['cows'] > 0:
                color = (0, 255, 255)  # Yellow
            elif "Calves:" in line and elk_types['calves'] > 0:
                color = (255, 165, 0)  # Orange
                
            cv2.putText(frame, line, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 12
        
        # Motion and quality info
        quality_lines = [
            f"Motion: {'Yes' if elk_results['motion_detected'] else 'No'}",
            f"Blur Score: {elk_results['blur_score']:.1f}",
            f"Quality: {'Good' if elk_results['blur_score'] > 100 else 'Blurry'}",
            "",
            f"Motion Comp: {'ON' if self.show_motion_compensation else 'OFF'}",
            f"Features: {'ON' if self.show_feature_analysis else 'OFF'}",
            f"Masks: {'ON' if self.show_masks else 'OFF'}",
            f"Tracking: {'ON' if self.show_tracking else 'OFF'}"
        ]
        
        y_offset = 30
        for line in quality_lines:
            color = (255, 255, 255)
            if "Motion: Yes" in line:
                color = (0, 255, 255)
            elif "Quality: Good" in line:
                color = (0, 255, 0)
            elif "Quality: Blurry" in line:
                color = (0, 165, 255)
                
            cv2.putText(frame, line, (quality_panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 15
        
        # Tracking info
        tracker_info = self.tracker.get_tracker_info()
        tracking_lines = [
            f"Distance Function: {tracker_info['distance_function']}",
            f"Distance Threshold: {tracker_info['distance_threshold']}",
            f"Active Tracks: {tracker_info['active_tracks']}",
            f"Frame Count: {tracker_info['frame_count']}"
        ]
        
        y_offset = tracking_panel_y + 20
        for line in tracking_lines:
            cv2.putText(frame, line, (quality_panel_x + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 15
        
        return frame
    
    def update_statistics(self, elk_results: dict, tracks: list, frame_num: int):
        """Update elk counting and tracking statistics."""
        elk_types = elk_results['elk_types']
        
        # Update maximum counts
        self.max_elk_counts['bulls'] = max(self.max_elk_counts['bulls'], elk_types['bulls'])
        self.max_elk_counts['cows'] = max(self.max_elk_counts['cows'], elk_types['cows'])
        self.max_elk_counts['calves'] = max(self.max_elk_counts['calves'], elk_types['calves'])
        self.max_elk_counts['total'] = max(self.max_elk_counts['total'], elk_results['total_elk_count'])
        
        # Store frame data for history
        frame_data = {
            'total': elk_results['total_elk_count'],
            'bulls': elk_types['bulls'],
            'cows': elk_types['cows'],
            'calves': elk_types['calves'],
            'motion_detected': elk_results['motion_detected'],
            'blur_score': elk_results['blur_score'],
            'active_tracks': len([t for t in tracks if t.get('is_confirmed', False)])
        }
        
        self.elk_history.append(frame_data)
        
        # Update track manager
        self.track_manager.add_frame_data(frame_num, tracks, elk_results['elk_segmentations'])
        
        # Keep only last 100 frames of history
        if len(self.elk_history) > 100:
            self.elk_history.pop(0)
    
    def run(self):
        """Run the enhanced elk segmentation player."""
        # Open video
        print(f"Opening video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")
        print("\nStarting Enhanced SAM2 Elk Segmentation...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  M: Toggle Motion Compensation")
        print("  F: Toggle Feature Analysis")
        print("  S: Toggle Masks")
        print("  T: Toggle Tracking")
        print("  Q/ESC: Quit")
        
        # Create window
        cv2.namedWindow('Enhanced SAM2 Elk Segmentation', cv2.WINDOW_RESIZABLE)
        
        # Playback variables
        frame_delay = 1.0 / fps
        paused = False
        frame_num = 0
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached.")
                    break
                
                frame_num += 1
                start_time = time.time()
                
                # Run enhanced elk segmentation
                elk_results = self.segmenter.segment_elk(
                    frame,
                    apply_motion_compensation=self.show_motion_compensation,
                    analyze_features=self.show_feature_analysis
                )
                
                # Run tracking if enabled
                tracks = []
                if self.show_tracking:
                    tracks = self.tracker.update(elk_results['elk_segmentations'], frame)
                
                # Update statistics
                self.update_statistics(elk_results, tracks, frame_num)
                
                # Draw segmentations
                annotated_frame = self.draw_elk_segmentations(frame, elk_results)
                
                # Draw tracking results
                if self.show_tracking:
                    annotated_frame = self.draw_tracking_results(annotated_frame, tracks)
                
                # Calculate processing FPS
                processing_time = time.time() - start_time
                processing_fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Add info panel
                final_frame = self.create_info_panel(
                    annotated_frame, elk_results, tracks, frame_num, frame_count, processing_fps
                )
                
                # Display frame
                cv2.imshow('Enhanced SAM2 Elk Segmentation', final_frame)
                
                # Calculate delay to maintain video FPS
                elapsed = time.time() - start_time
                delay = max(0, frame_delay - elapsed)
                
            else:
                delay = 0.03  # 30ms delay when paused
            
            # Handle key presses
            key = cv2.waitKey(int(delay * 1000)) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('m'):  # Toggle motion compensation
                self.show_motion_compensation = not self.show_motion_compensation
                print(f"Motion compensation: {'ON' if self.show_motion_compensation else 'OFF'}")
            elif key == ord('f'):  # Toggle feature analysis
                self.show_feature_analysis = not self.show_feature_analysis
                print(f"Feature analysis: {'ON' if self.show_feature_analysis else 'OFF'}")
            elif key == ord('s'):  # Toggle masks
                self.show_masks = not self.show_masks
                print(f"Masks: {'ON' if self.show_masks else 'OFF'}")
            elif key == ord('t'):  # Toggle tracking
                self.show_tracking = not self.show_tracking
                print(f"Tracking: {'ON' if self.show_tracking else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.segmenter.cleanup()
        
        # Print final statistics
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print comprehensive final statistics."""
        print(f"\n{'='*70}")
        print("ENHANCED SAM2 ELK SEGMENTATION SUMMARY")
        print(f"{'='*70}")
        print(f"Maximum elk counts observed:")
        print(f"  Bulls: {self.max_elk_counts['bulls']}")
        print(f"  Cows: {self.max_elk_counts['cows']}")
        print(f"  Calves: {self.max_elk_counts['calves']}")
        print(f"  Total: {self.max_elk_counts['total']}")
        
        if self.elk_history:
            # Calculate averages
            avg_total = sum(frame['total'] for frame in self.elk_history) / len(self.elk_history)
            avg_bulls = sum(frame['bulls'] for frame in self.elk_history) / len(self.elk_history)
            avg_cows = sum(frame['cows'] for frame in self.elk_history) / len(self.elk_history)
            avg_calves = sum(frame['calves'] for frame in self.elk_history) / len(self.elk_history)
            
            frames_with_motion = sum(1 for frame in self.elk_history if frame['motion_detected'])
            avg_blur = sum(frame['blur_score'] for frame in self.elk_history) / len(self.elk_history)
            avg_tracks = sum(frame['active_tracks'] for frame in self.elk_history) / len(self.elk_history)
            
            print(f"\nAverage elk per frame:")
            print(f"  Bulls: {avg_bulls:.1f}")
            print(f"  Cows: {avg_cows:.1f}")
            print(f"  Calves: {avg_calves:.1f}")
            print(f"  Total: {avg_total:.1f}")
            
            print(f"\nVideo quality metrics:")
            print(f"  Frames with motion: {frames_with_motion}/{len(self.elk_history)} ({frames_with_motion/len(self.elk_history)*100:.1f}%)")
            print(f"  Average blur score: {avg_blur:.1f}")
            
            print(f"\nTracking metrics:")
            print(f"  Average active tracks: {avg_tracks:.1f}")
        
        # Print track manager summary
        print(f"\n{self.track_manager.summary_report()}")
        
        print(f"{'='*70}")


def main():
    """Main function to run the enhanced elk segmenter."""
    # Configuration
    VIDEO_PATH = "data/raw/grassland.mp4"
    CONFIG_PATH = None  # Use default config
    CONFIDENCE_THRESHOLD = 0.5
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        VIDEO_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        CONFIG_PATH = sys.argv[2]
    if len(sys.argv) > 3:
        CONFIDENCE_THRESHOLD = float(sys.argv[3])
    
    # Check if video file exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        print("Available videos:")
        data_dir = Path("data/raw")
        if data_dir.exists():
            for video_file in data_dir.glob("*.mp4"):
                print(f"  {video_file}")
        sys.exit(1)
    
    try:
        # Create and run enhanced segmenter
        player = EnhancedElkSegmentationPlayer(
            video_path=VIDEO_PATH,
            config_path=CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        player.run()
        
    except KeyboardInterrupt:
        print("\nSegmentation interrupted by user.")
    except Exception as e:
        print(f"Error during segmentation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
