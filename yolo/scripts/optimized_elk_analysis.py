#!/usr/bin/env python3
"""
Optimized Elk Analysis Script

This script processes the grassland.mp4 video with optimized settings for maximum
elk detection, focusing on accurate elk counting and bull/cow classification.
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
from src.detection.optimized_elk_detector import OptimizedElkDetector


def optimized_elk_analysis(video_path, model_path, output_dir, confidence_threshold=0.15):
    """
    Run optimized elk analysis for maximum elk detection.
    
    Args:
        video_path: Path to the video file
        model_path: Path to YOLO model
        output_dir: Directory to save results
        confidence_threshold: Very low threshold for maximum detection
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "optimized_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize optimized elk detector
    print("Initializing Optimized Elk Detector...")
    detector = OptimizedElkDetector(
        model_path=model_path,
        device="auto",
        confidence_threshold=confidence_threshold,
        iou_threshold=0.4  # Lower to avoid merging separate elk
    )
    
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
    
    # Processing parameters - analyze more frames for better detection
    FRAME_SKIP = max(1, int(fps // 5))  # Sample ~5 frames per second (more than before)
    print(f"Processing every {FRAME_SKIP} frames for maximum elk detection...")
    
    # Colors for elk types
    colors = {
        'bull': (0, 255, 0),    # Green for bulls
        'cow': (0, 255, 255),   # Yellow for cows (includes calves)
    }
    
    # Process video
    detection_results = []
    frame_data = []
    frame_idx = 0
    processed_frames = 0
    saved_frames = 0
    
    # Track maximum elk counts
    max_elk_in_frame = 0
    max_bulls_in_frame = 0
    max_cows_in_frame = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for processing
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        
        processed_frames += 1
        
        # Run optimized elk detection
        elk_results = detector.detect_elk(frame, apply_motion_compensation=True)
        
        elk_count = elk_results['total_elk_count']
        elk_types = elk_results['elk_types']
        
        # Update maximum counts
        max_elk_in_frame = max(max_elk_in_frame, elk_count)
        max_bulls_in_frame = max(max_bulls_in_frame, elk_types['bulls'])
        max_cows_in_frame = max(max_cows_in_frame, elk_types['cows'])
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw elk detections with proper labeling
        for elk_detection in elk_results['elk_detections']:
            bbox = elk_detection['bbox']
            elk_type = elk_detection['elk_type']
            confidence = elk_detection['confidence']
            elk_confidence = elk_detection['elk_confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[elk_type]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
            
            # Create proper elk label (no animal names, only "Elk")
            label = f"Elk {elk_type.title()}: {elk_confidence:.2f}"
            
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
            
            # Store detection data
            detection_results.append({
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'elk_confidence': float(elk_confidence),
                'elk_type': elk_type,
                'area': float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            })
        
        # Add frame info overlay
        info_text = f"Frame: {frame_idx} | Time: {frame_idx/fps:.1f}s | Elk: {elk_count} (Bulls: {elk_types['bulls']}, Cows: {elk_types['cows']})"
        cv2.putText(
            annotated_frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )
        
        # Save frame if elk detected or every 10th processed frame
        if elk_count > 0 or processed_frames % 10 == 0:
            frame_filename = f"frame_{frame_idx:06d}_elk_{elk_count}_bulls_{elk_types['bulls']}_cows_{elk_types['cows']}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, annotated_frame)
            saved_frames += 1
            
            if elk_count > 0:
                print(f"Frame {frame_idx}: Found {elk_count} elk ({elk_types['bulls']} bulls, {elk_types['cows']} cows) - saved {frame_filename}")
        
        # Store frame data
        frame_info = {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'total_elk': elk_count,
            'bulls': elk_types['bulls'],
            'cows': elk_types['cows'],
            'motion_detected': elk_results['motion_detected'],
            'blur_score': elk_results['blur_score']
        }
        
        frame_data.append(frame_info)
        frame_idx += 1
        
        # Progress update
        if processed_frames % 25 == 0:
            print(f"Processed {processed_frames} frames, found {len(detection_results)} elk detections")
    
    cap.release()
    
    # Analysis results
    print(f"\nOptimized Analysis Complete!")
    print(f"Processed {processed_frames} frames")
    print(f"Found {len(detection_results)} total elk detections")
    print(f"Saved {saved_frames} annotated frames")
    
    # Calculate comprehensive elk statistics
    if detection_results:
        # Frame-based statistics
        frames_with_elk = [f for f in frame_data if f['total_elk'] > 0]
        
        # Bull/cow statistics
        bull_detections = [d for d in detection_results if d['elk_type'] == 'bull']
        cow_detections = [d for d in detection_results if d['elk_type'] == 'cow']
        
        # Confidence statistics
        high_conf_detections = [d for d in detection_results if d['elk_confidence'] > 0.5]
        
        print(f"\nElk Population Analysis:")
        print(f"  Maximum elk in single frame: {max_elk_in_frame}")
        print(f"  Maximum bulls in single frame: {max_bulls_in_frame}")
        print(f"  Maximum cows in single frame: {max_cows_in_frame}")
        print(f"  Frames with elk: {len(frames_with_elk)}/{processed_frames} ({len(frames_with_elk)/processed_frames*100:.1f}%)")
        
        if frames_with_elk:
            avg_elk_per_frame = sum(f['total_elk'] for f in frames_with_elk) / len(frames_with_elk)
            avg_bulls_per_frame = sum(f['bulls'] for f in frames_with_elk) / len(frames_with_elk)
            avg_cows_per_frame = sum(f['cows'] for f in frames_with_elk) / len(frames_with_elk)
            
            print(f"  Average elk per frame (with elk): {avg_elk_per_frame:.1f}")
            print(f"  Average bulls per frame (with elk): {avg_bulls_per_frame:.1f}")
            print(f"  Average cows per frame (with elk): {avg_cows_per_frame:.1f}")
        
        print(f"\nDetection Breakdown:")
        print(f"  Total bull detections: {len(bull_detections)}")
        print(f"  Total cow detections: {len(cow_detections)} (includes calves)")
        print(f"  High confidence detections: {len(high_conf_detections)}")
        
        # Estimated elk count (most conservative approach)
        estimated_elk_count = max_elk_in_frame
        estimated_bull_count = max_bulls_in_frame
        estimated_cow_count = max_cows_in_frame
        
        print(f"\n*** ESTIMATED ELK POPULATION ***")
        print(f"Total Elk: {estimated_elk_count}")
        print(f"Bulls: {estimated_bull_count}")
        print(f"Cows (including calves): {estimated_cow_count}")
        print(f"Bull/Cow Ratio: {estimated_bull_count/estimated_cow_count:.2f}" if estimated_cow_count > 0 else "Bull/Cow Ratio: N/A")
        
    else:
        print("No elk detected in the video.")
        estimated_elk_count = 0
        estimated_bull_count = 0
        estimated_cow_count = 0
    
    # Save comprehensive results
    summary_report = {
        'analysis_info': {
            'video_file': video_path,
            'analysis_date': datetime.now().isoformat(),
            'model_used': model_path,
            'confidence_threshold': confidence_threshold,
            'analysis_type': 'optimized_maximum_detection'
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
            'frames_saved': saved_frames,
            'frames_with_elk': len([f for f in frame_data if f['total_elk'] > 0])
        },
        'elk_population_estimate': {
            'total_elk': estimated_elk_count,
            'bulls': estimated_bull_count,
            'cows_including_calves': estimated_cow_count,
            'bull_cow_ratio': estimated_bull_count/estimated_cow_count if estimated_cow_count > 0 else None
        },
        'detection_statistics': {
            'total_detections': len(detection_results),
            'bull_detections': len([d for d in detection_results if d['elk_type'] == 'bull']),
            'cow_detections': len([d for d in detection_results if d['elk_type'] == 'cow']),
            'high_confidence_detections': len([d for d in detection_results if d['elk_confidence'] > 0.5]),
            'max_elk_single_frame': max_elk_in_frame,
            'max_bulls_single_frame': max_bulls_in_frame,
            'max_cows_single_frame': max_cows_in_frame
        },
        'detailed_detections': detection_results[:200]  # Save first 200 detections
    }
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, "optimized_elk_analysis.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Annotated frames: {frames_dir}")
    
    return summary_report


def main():
    """Main function to run optimized elk analysis."""
    # Configuration
    VIDEO_PATH = "data/raw/grassland.mp4"
    MODEL_PATH = "yolov8n.pt"
    OUTPUT_DIR = "results/optimized_elk_analysis"
    CONFIDENCE_THRESHOLD = 0.15  # Very low for maximum detection
    
    # Check if files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    try:
        print("="*70)
        print("OPTIMIZED ELK ANALYSIS - MAXIMUM DETECTION")
        print("="*70)
        
        # Run optimized analysis
        results = optimized_elk_analysis(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            confidence_threshold=CONFIDENCE_THRESHOLD
        )
        
        print("\n" + "="*70)
        print("OPTIMIZED ANALYSIS COMPLETE")
        print("="*70)
        print(f"Estimated Total Elk: {results['elk_population_estimate']['total_elk']}")
        print(f"Estimated Bulls: {results['elk_population_estimate']['bulls']}")
        print(f"Estimated Cows (inc. calves): {results['elk_population_estimate']['cows_including_calves']}")
        print(f"Total Detections: {results['detection_statistics']['total_detections']}")
        print(f"Frames Processed: {results['processing_info']['frames_processed']}")
        print(f"Frames with Elk: {results['processing_info']['frames_with_elk']}")
        print("="*70)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
