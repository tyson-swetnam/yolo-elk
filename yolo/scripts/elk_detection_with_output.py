#!/usr/bin/env python3
"""
Elk Detection with Output Script

This script processes the grassland.mp4 video, detects elk using YOLO,
and saves annotated frames showing the detections.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np
import json
from datetime import datetime
from src.detection.yolo_detector import YOLODetector


def process_elk_video_with_output(video_path, model_path, output_dir, confidence_threshold=0.3):
    """
    Process video for elk detection and save annotated frames.
    
    Args:
        video_path: Path to the video file
        model_path: Path to YOLO model
        output_dir: Directory to save results
        confidence_threshold: Minimum confidence for detections
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "annotated_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize YOLO detector
    print("Initializing YOLO detector...")
    detector = YOLODetector(
        model_path=model_path,
        device="auto",
        confidence_threshold=confidence_threshold,
        iou_threshold=0.5
    )
    
    # Define potential elk/animal classes
    elk_classes = [17, 18, 19, 20, 21, 22, 23]  # Large mammals in COCO
    elk_keywords = ['horse', 'cow', 'sheep', 'deer', 'elk', 'animal', 'bear', 'zebra']
    
    # Open video
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    
    print(f"Video properties: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s, {frame_count} frames")
    
    # Processing parameters
    FRAME_SKIP = max(1, int(fps // 2))  # Sample ~2 frames per second
    print(f"Processing every {FRAME_SKIP} frames...")
    
    # Colors for different confidence levels
    colors = {
        'high': (0, 255, 0),    # Green for high confidence (>0.7)
        'medium': (0, 255, 255), # Yellow for medium confidence (0.5-0.7)
        'low': (0, 165, 255)     # Orange for low confidence (<0.5)
    }
    
    def get_box_color(confidence):
        if confidence > 0.7:
            return colors['high']
        elif confidence > 0.5:
            return colors['medium']
        else:
            return colors['low']
    
    def is_potential_elk(class_id, class_name):
        if class_id in elk_classes:
            return True
        class_name_lower = class_name.lower()
        return any(keyword in class_name_lower for keyword in elk_keywords)
    
    # Process video
    detection_results = []
    frame_data = []
    frame_idx = 0
    processed_frames = 0
    saved_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for faster processing
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        
        processed_frames += 1
        
        # Run YOLO detection
        detections = detector.detect(frame)
        
        # Filter for potential elk detections
        elk_detections = []
        elk_count = 0
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        for i in range(len(detections['boxes'])):
            class_id = detections['class_ids'][i]
            class_name = detections['class_names'][i]
            confidence = detections['scores'][i]
            box = detections['boxes'][i]
            
            # Check if this could be an elk
            if is_potential_elk(class_id, class_name):
                elk_count += 1
                
                # Store detection data
                elk_detections.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / fps,
                    'bbox': box.tolist(),
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'area': float((box[2] - box[0]) * (box[3] - box[1]))
                })
                
                # Draw bounding box on frame
                x1, y1, x2, y2 = map(int, box)
                color = get_box_color(confidence)
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Prepare label
                label = f"ELK ({class_name}): {confidence:.2f}"
                
                # Get text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                
                # Draw label background
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - 15),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 0),
                    2
                )
        
        # Add frame info overlay
        info_text = f"Frame: {frame_idx} | Time: {frame_idx/fps:.1f}s | Elk: {elk_count}"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        
        # Save frame if elk detected or every 10th processed frame
        if elk_count > 0 or processed_frames % 10 == 0:
            frame_filename = f"frame_{frame_idx:06d}_elk_{elk_count}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, annotated_frame)
            saved_frames += 1
            
            if elk_count > 0:
                print(f"Frame {frame_idx}: Found {elk_count} elk - saved {frame_filename}")
        
        # Store frame data
        frame_info = {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'total_detections': len(detections['boxes']),
            'elk_detections': elk_count,
            'saved_frame': elk_count > 0 or processed_frames % 10 == 0
        }
        
        frame_data.append(frame_info)
        detection_results.extend(elk_detections)
        
        frame_idx += 1
        
        # Progress update
        if processed_frames % 20 == 0:
            print(f"Processed {processed_frames} frames, found {len(detection_results)} elk detections")
    
    cap.release()
    
    # Analysis results
    print(f"\nProcessing complete!")
    print(f"Processed {processed_frames} frames")
    print(f"Found {len(detection_results)} elk detections")
    print(f"Saved {saved_frames} annotated frames")
    
    # Calculate elk count estimates
    if detection_results:
        # Maximum detections in any single frame
        max_elk_single_frame = max(frame['elk_detections'] for frame in frame_data)
        
        # Average detections across frames with detections
        frames_with_elk = [f for f in frame_data if f['elk_detections'] > 0]
        avg_elk_per_frame = sum(f['elk_detections'] for f in frames_with_elk) / len(frames_with_elk) if frames_with_elk else 0
        
        # High confidence detections
        high_conf_detections = [d for d in detection_results if d['confidence'] > 0.5]
        
        print(f"\nElk Count Estimates:")
        print(f"  Maximum in single frame: {max_elk_single_frame}")
        print(f"  Average per frame with elk: {avg_elk_per_frame:.1f}")
        print(f"  High confidence detections: {len(high_conf_detections)}")
        print(f"  Frames with elk: {len(frames_with_elk)}")
        
        estimated_elk_count = max_elk_single_frame
        print(f"\n*** ESTIMATED ELK COUNT: {estimated_elk_count} ***")
        
        # Class distribution
        class_counts = {}
        for det in detection_results:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"\nDetected classes:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count}")
    else:
        print("No elk detected in the video.")
        estimated_elk_count = 0
    
    # Save detailed results
    summary_report = {
        'analysis_info': {
            'video_file': video_path,
            'analysis_date': datetime.now().isoformat(),
            'model_used': model_path,
            'confidence_threshold': confidence_threshold
        },
        'video_properties': {
            'duration_seconds': duration,
            'total_frames': frame_count,
            'fps': fps,
            'resolution': f"{width}x{height}"
        },
        'processing_info': {
            'frames_processed': processed_frames,
            'frame_skip': FRAME_SKIP,
            'frames_saved': saved_frames
        },
        'detection_results': {
            'total_detections': len(detection_results),
            'estimated_elk_count': estimated_elk_count,
            'max_elk_single_frame': max_elk_single_frame if detection_results else 0,
            'frames_with_elk': len([f for f in frame_data if f['elk_detections'] > 0]),
            'class_distribution': class_counts if detection_results else {}
        },
        'detailed_detections': detection_results[:100]  # Save first 100 detections
    }
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, "elk_detection_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Annotated frames: {frames_dir}")
    
    return summary_report


def main():
    """Main function to run elk detection with output."""
    # Configuration
    VIDEO_PATH = "data/raw/grassland.mp4"
    MODEL_PATH = "yolov8n.pt"
    OUTPUT_DIR = "results/elk_detection_output"
    CONFIDENCE_THRESHOLD = 0.3
    
    # Check if files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    try:
        print("="*60)
        print("ELK DETECTION WITH OUTPUT")
        print("="*60)
        
        # Run analysis
        results = process_elk_video_with_output(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Estimated elk count: {results['detection_results']['estimated_elk_count']}")
        print(f"Total detections: {results['detection_results']['total_detections']}")
        print(f"Frames processed: {results['processing_info']['frames_processed']}")
        print(f"Frames saved: {results['processing_info']['frames_saved']}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
