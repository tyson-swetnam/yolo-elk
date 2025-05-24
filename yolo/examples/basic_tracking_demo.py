#!/usr/bin/env python3
"""
Basic YOLO-ELK Tracking Demo

This script demonstrates basic object detection and tracking using the YOLO-ELK system.
It can process video files, webcam input, or image sequences.
"""

import sys
import os
import cv2
import numpy as np
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from detection.yolo_detector import YOLODetector
from tracking.norfair_tracker import NorfairTracker


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLO-ELK Tracking Demo")
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="0",
        help="Input source: video file path, webcam index (0), or image directory"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video file path (optional)"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path"
    )
    
    parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Target classes to track (e.g., person car)"
    )
    
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=30,
        help="Tracking distance threshold"
    )
    
    parser.add_argument(
        "--max-age",
        type=int,
        default=15,
        help="Maximum frames to keep track without detection"
    )
    
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable display window"
    )
    
    return parser.parse_args()


def setup_video_capture(input_source):
    """Setup video capture from various sources."""
    if input_source.isdigit():
        # Webcam
        cap = cv2.VideoCapture(int(input_source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam {input_source}")
    elif os.path.isfile(input_source):
        # Video file
        cap = cv2.VideoCapture(input_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {input_source}")
    else:
        raise ValueError(f"Invalid input source: {input_source}")
    
    return cap


def setup_video_writer(output_path, cap, fps=30):
    """Setup video writer for output."""
    if output_path is None:
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return out


def draw_detections(frame, detections, class_names=None):
    """Draw detection bounding boxes on frame."""
    for det in detections:
        x1, y1, x2, y2, score = det[:5]
        class_id = int(det[5]) if len(det) > 5 else 0
        
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        # Draw label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {score:.2f}"
        else:
            label = f"Class {class_id}: {score:.2f}"
        
        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


def draw_tracks(frame, tracks, class_names=None):
    """Draw tracking results on frame."""
    for track in tracks:
        if not track['is_confirmed']:
            continue
            
        bbox = track['bbox']
        track_id = track['track_id']
        class_id = track['class_id']
        confidence = track['confidence']
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw track ID
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw class label
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Class {class_id}: {confidence:.2f}"
        
        cv2.putText(frame, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw center point
        center = track['center']
        cv2.circle(frame, (int(center[0]), int(center[1])), 3, (255, 0, 0), -1)
    
    return frame


def main():
    """Main function."""
    args = parse_arguments()
    
    print("🚀 YOLO-ELK Tracking Demo")
    print("=" * 40)
    
    # Initialize detector
    print(f"Loading YOLO model: {args.model}")
    detector = YOLODetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        target_classes=None  # Will filter by class names later if specified
    )
    
    # Get class names and filter target classes
    class_names = detector.class_names
    target_class_ids = None
    
    if args.classes:
        # Convert class names to IDs
        name_to_id = {name: id for id, name in class_names.items()}
        target_class_ids = []
        for class_name in args.classes:
            if class_name in name_to_id:
                target_class_ids.append(name_to_id[class_name])
            else:
                print(f"Warning: Class '{class_name}' not found in model")
        
        if target_class_ids:
            detector.target_classes = target_class_ids
            print(f"Tracking classes: {args.classes}")
    
    # Initialize tracker
    print("Initializing tracker...")
    tracker = NorfairTracker(
        distance_function="euclidean",
        distance_threshold=args.distance_threshold,
        hit_counter_max=args.max_age,
        detection_threshold=args.confidence
    )
    
    # Setup video capture
    print(f"Setting up input: {args.input}")
    cap = setup_video_capture(args.input)
    
    # Setup video writer
    out = setup_video_writer(args.output, cap) if args.output else None
    if args.output:
        print(f"Output will be saved to: {args.output}")
    
    # Processing loop
    frame_count = 0
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    
    print("\nStarting processing... Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run detection
            detections = detector.detect_and_track_format(frame)
            
            # Update tracker
            tracks = tracker.update(detections, frame)
            
            # Draw results
            if not args.no_display or out is not None:
                # Draw detections (green)
                frame_vis = draw_detections(frame.copy(), detections, class_names)
                
                # Draw tracks (red)
                frame_vis = draw_tracks(frame_vis, tracks, class_names)
                
                # Draw info
                info_text = [
                    f"Frame: {frame_count}",
                    f"Detections: {len(detections)}",
                    f"Active Tracks: {len([t for t in tracks if t['is_confirmed']])}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(frame_vis, text, (10, 30 + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Calculate and display FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = cv2.getTickCount()
                    fps = 30 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                    fps_start_time = fps_end_time
                    
                cv2.putText(frame_vis, f"FPS: {fps:.1f}", (10, frame_vis.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Save frame
                if out is not None:
                    out.write(frame_vis)
                
                # Display frame
                if not args.no_display:
                    cv2.imshow('YOLO-ELK Tracking', frame_vis)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Reset tracker
                        tracker.reset()
                        print("Tracker reset")
            
            # Print progress
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\nProcessing completed!")
        print(f"Total frames processed: {frame_count}")
        print(f"Final active tracks: {tracker.get_track_count()}")


if __name__ == "__main__":
    main()
