#!/usr/bin/env python3
"""
Enhanced Real-time Elk Detection Script

This script opens the grassland.mp4 video and uses the enhanced ElkDetector
to identify and classify elk (bulls, cows, calves) in real-time with motion
compensation for helicopter footage.
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
from src.detection.elk_detector import ElkDetector


class EnhancedElkPlayer:
    """Enhanced real-time elk detection with helicopter motion compensation."""
    
    def __init__(self, video_path, model_path, confidence_threshold=0.25):
        """
        Initialize the enhanced elk detector.
        
        Args:
            video_path: Path to the video file
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.video_path = video_path
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize enhanced elk detector
        print("Initializing Enhanced Elk Detector...")
        self.detector = ElkDetector(
            model_path=model_path,
            device="auto",
            confidence_threshold=confidence_threshold,
            iou_threshold=0.5
        )
        
        # Tracking and statistics
        self.elk_history = []
        self.max_elk_counts = {'bulls': 0, 'cows': 0, 'calves': 0, 'total': 0}
        
        # Display settings
        self.show_motion_compensation = True
        self.show_feature_analysis = True
        self.show_confidence_bars = True
        
    def draw_elk_detection(self, frame: np.ndarray, elk_detection: dict) -> np.ndarray:
        """
        Draw a single elk detection with enhanced visualization.
        
        Args:
            frame: Input frame
            elk_detection: Elk detection dictionary
            
        Returns:
            Frame with drawn detection
        """
        bbox = elk_detection['bbox']
        elk_type = elk_detection['elk_type']
        confidence = elk_detection['confidence']
        elk_confidence = elk_detection['elk_confidence']
        features = elk_detection['features']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Get color for elk type
        color = self.detector.get_elk_color(elk_type)
        
        # Draw bounding box with thickness based on confidence
        thickness = max(2, int(confidence * 5))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare detailed label
        label_lines = [
            f"{elk_type.upper()}: {elk_confidence:.2f}",
            f"YOLO: {confidence:.2f}"
        ]
        
        # Add feature information if available
        if self.show_feature_analysis and features:
            if features.get('has_elk_colors', False):
                label_lines.append("✓ Elk Colors")
            if features.get('white_rump_score', 0) > 0.1:
                label_lines.append("✓ White Rump")
        
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
        label_y = y1 - total_height - 10
        if label_y < 0:
            label_y = y2 + 10
        
        cv2.rectangle(
            frame,
            (x1, label_y),
            (x1 + max_width + 10, label_y + total_height + 10),
            color,
            -1
        )
        
        # Draw label text
        current_y = label_y + 20
        for i, line in enumerate(label_lines):
            cv2.putText(
                frame,
                line,
                (x1 + 5, current_y),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness
            )
            current_y += line_heights[i] + 5
        
        # Draw confidence bar if enabled
        if self.show_confidence_bars:
            bar_width = x2 - x1
            bar_height = 8
            bar_y = y2 + 5
            
            # Background bar
            cv2.rectangle(frame, (x1, bar_y), (x2, bar_y + bar_height), (50, 50, 50), -1)
            
            # Confidence bar
            conf_width = int(bar_width * elk_confidence)
            cv2.rectangle(frame, (x1, bar_y), (x1 + conf_width, bar_y + bar_height), color, -1)
        
        return frame
    
    def draw_elk_detections(self, frame: np.ndarray, elk_results: dict) -> np.ndarray:
        """
        Draw all elk detections on the frame.
        
        Args:
            frame: Input frame
            elk_results: Results from elk detector
            
        Returns:
            Frame with all detections drawn
        """
        annotated_frame = frame.copy()
        
        # Draw each elk detection
        for elk_detection in elk_results['elk_detections']:
            annotated_frame = self.draw_elk_detection(annotated_frame, elk_detection)
        
        return annotated_frame
    
    def create_info_panel(self, frame: np.ndarray, elk_results: dict, frame_num: int, 
                         total_frames: int, fps: float) -> np.ndarray:
        """
        Create comprehensive information panel.
        
        Args:
            frame: Input frame
            elk_results: Results from elk detector
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
        panel_height = 200
        
        cv2.rectangle(overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        
        # Elk count panel
        count_panel_y = panel_height + 20
        count_panel_height = 120
        cv2.rectangle(overlay, (10, count_panel_y), (panel_width, count_panel_y + count_panel_height), (0, 0, 0), -1)
        
        # Motion/quality panel
        quality_panel_x = panel_width + 20
        quality_panel_width = 250
        quality_panel_height = 100
        cv2.rectangle(overlay, (quality_panel_x, 10), (quality_panel_x + quality_panel_width, quality_panel_height), (0, 0, 0), -1)
        
        # Blend with original frame
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Main info text
        info_lines = [
            f"Frame: {frame_num}/{total_frames}",
            f"Progress: {frame_num/total_frames*100:.1f}%",
            f"Processing FPS: {fps:.1f}",
            f"Total Elk: {elk_results['total_elk_count']}",
            "",
            "Controls:",
            "SPACE: Pause/Resume",
            "M: Toggle Motion Compensation",
            "F: Toggle Feature Analysis", 
            "C: Toggle Confidence Bars",
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
            f"Calves: {self.max_elk_counts['calves']}"
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
            f"Features: {'ON' if self.show_feature_analysis else 'OFF'}"
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
        
        return frame
    
    def update_statistics(self, elk_results: dict):
        """Update elk counting statistics."""
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
            'blur_score': elk_results['blur_score']
        }
        
        self.elk_history.append(frame_data)
        
        # Keep only last 100 frames of history
        if len(self.elk_history) > 100:
            self.elk_history.pop(0)
    
    def run(self):
        """Run the enhanced elk detection player."""
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
        print("\nStarting Enhanced Elk Detection...")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  M: Toggle Motion Compensation")
        print("  F: Toggle Feature Analysis")
        print("  C: Toggle Confidence Bars")
        print("  Q/ESC: Quit")
        
        # Create window
        cv2.namedWindow('Enhanced Elk Detection', cv2.WINDOW_RESIZABLE)
        
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
                
                # Run enhanced elk detection
                elk_results = self.detector.detect_elk(
                    frame,
                    apply_motion_compensation=self.show_motion_compensation,
                    analyze_features=self.show_feature_analysis
                )
                
                # Update statistics
                self.update_statistics(elk_results)
                
                # Draw elk detections
                annotated_frame = self.draw_elk_detections(frame, elk_results)
                
                # Calculate processing FPS
                processing_time = time.time() - start_time
                processing_fps = 1.0 / processing_time if processing_time > 0 else 0
                
                # Add info panel
                final_frame = self.create_info_panel(
                    annotated_frame, elk_results, frame_num, frame_count, processing_fps
                )
                
                # Display frame
                cv2.imshow('Enhanced Elk Detection', final_frame)
                
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
            elif key == ord('c'):  # Toggle confidence bars
                self.show_confidence_bars = not self.show_confidence_bars
                print(f"Confidence bars: {'ON' if self.show_confidence_bars else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print comprehensive final statistics."""
        print(f"\n{'='*60}")
        print("ENHANCED ELK DETECTION SUMMARY")
        print(f"{'='*60}")
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
            
            print(f"\nAverage elk per frame:")
            print(f"  Bulls: {avg_bulls:.1f}")
            print(f"  Cows: {avg_cows:.1f}")
            print(f"  Calves: {avg_calves:.1f}")
            print(f"  Total: {avg_total:.1f}")
            
            print(f"\nVideo quality metrics:")
            print(f"  Frames with motion: {frames_with_motion}/{len(self.elk_history)} ({frames_with_motion/len(self.elk_history)*100:.1f}%)")
            print(f"  Average blur score: {avg_blur:.1f}")
        
        print(f"{'='*60}")


def main():
    """Main function to run the enhanced elk detector."""
    # Configuration
    VIDEO_PATH = "data/raw/grassland.mp4"
    MODEL_PATH = "yolov8n.pt"
    CONFIDENCE_THRESHOLD = 0.25  # Lower for motion blur
    
    # Check if files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    try:
        # Create and run enhanced detector
        player = EnhancedElkPlayer(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        player.run()
        
    except KeyboardInterrupt:
        print("\nDetection interrupted by user.")
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
