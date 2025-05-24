#!/usr/bin/env python3
"""
Dynamic Elk Analysis Script

This script processes the grassland.mp4 video with dynamic adaptation to changing
elk sizes and helicopter movement, providing the most accurate elk counting possible.
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
from src.detection.dynamic_elk_detector import DynamicElkDetector


def dynamic_elk_analysis(video_path, model_path, output_dir, min_elk_size=50):
    """
    Run dynamic elk analysis with adaptive size thresholds and movement awareness.
    
    Args:
        video_path: Path to the video file
        model_path: Path to YOLO model
        output_dir: Directory to save results
        min_elk_size: Minimum elk size in pixels (very small for distant elk)
    """
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "dynamic_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Initialize dynamic elk detector
    print("Initializing Dynamic Elk Detector...")
    detector = DynamicElkDetector(
        model_path=model_path,
        device="auto",
        confidence_threshold=0.12,  # Very low base threshold
        iou_threshold=0.3,  # Lower to avoid merging separate elk
        min_elk_size=min_elk_size,  # Ultra-low minimum size
        history_length=15  # More history for better adaptation
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
    
    # Processing parameters - high frame rate for dynamic tracking
    FRAME_SKIP = max(1, int(fps // 10))  # Sample ~10 frames per second for maximum coverage
    print(f"Processing every {FRAME_SKIP} frames for dynamic elk tracking...")
    
    # Colors for elk types
    colors = {
        'bull': (0, 255, 0),    # Green for bulls
        'cow': (0, 255, 255),   # Yellow for cows
    }
    
    # Process video
    detection_results = []
    frame_data = []
    frame_idx = 0
    processed_frames = 0
    saved_frames = 0
    
    # Track maximum elk counts and dynamic statistics
    max_elk_in_frame = 0
    max_bulls_in_frame = 0
    max_cows_in_frame = 0
    
    # Dynamic tracking variables
    size_progression = []
    confidence_progression = []
    movement_events = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for processing
        if frame_idx % FRAME_SKIP != 0:
            frame_idx += 1
            continue
        
        processed_frames += 1
        
        # Run dynamic elk detection
        elk_results = detector.detect_elk_dynamic(frame, apply_motion_compensation=True)
        
        elk_count = elk_results['total_elk_count']
        elk_types = elk_results['elk_types']
        movement_info = elk_results['movement_info']
        size_analysis = elk_results['size_analysis']
        adaptive_confidence = elk_results['adaptive_confidence']
        
        # Update maximum counts
        max_elk_in_frame = max(max_elk_in_frame, elk_count)
        max_bulls_in_frame = max(max_bulls_in_frame, elk_types['bulls'])
        max_cows_in_frame = max(max_cows_in_frame, elk_types['cows'])
        
        # Track dynamic progression
        size_progression.append({
            'frame': frame_idx,
            'avg_size': size_analysis['avg_size'],
            'size_range': size_analysis['size_range'],
            'bull_threshold': size_analysis['bull_threshold']
        })
        
        confidence_progression.append({
            'frame': frame_idx,
            'adaptive_confidence': adaptive_confidence,
            'movement_detected': movement_info['movement_detected'],
            'movement_magnitude': movement_info['movement_magnitude']
        })
        
        # Track significant movement events
        if movement_info['movement_detected']:
            movement_events.append({
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'movement_info': movement_info
            })
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        # Draw elk detections with dynamic information
        for elk_detection in elk_results['elk_detections']:
            bbox = elk_detection['bbox']
            elk_type = elk_detection['elk_type']
            confidence = elk_detection['confidence']
            area = elk_detection['area']
            relative_size = elk_detection.get('relative_size', 1.0)
            size_rank = elk_detection.get('size_rank', 1)
            
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[elk_type]
            
            # Draw bounding box with thickness based on confidence
            thickness = max(2, int(confidence * 5))
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Create detailed elk label with dynamic info
            label = f"Elk {elk_type.title()}: {confidence:.2f}"
            size_info = f"Size: {int(area)}px (#{size_rank})"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            (size_width, size_text_height), _ = cv2.getTextSize(
                size_info, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw label background
            total_height = text_height + size_text_height + 20
            total_width = max(text_width, size_width) + 10
            
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - total_height),
                (x1 + total_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated_frame,
                label,
                (x1 + 5, y1 - size_text_height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
            
            # Draw size info
            cv2.putText(
                annotated_frame,
                size_info,
                (x1 + 5, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
            
            # Store detection data with dynamic information
            detection_results.append({
                'frame_idx': frame_idx,
                'timestamp': frame_idx / fps,
                'bbox': bbox.tolist(),
                'confidence': float(confidence),
                'elk_type': elk_type,
                'area': float(area),
                'relative_size': float(relative_size),
                'size_rank': int(size_rank),
                'adaptive_confidence_used': float(adaptive_confidence),
                'movement_detected': movement_info['movement_detected'],
                'bull_threshold_used': float(size_analysis['bull_threshold'])
            })
        
        # Add comprehensive frame info overlay
        info_lines = [
            f"Frame: {frame_idx} | Time: {frame_idx/fps:.1f}s",
            f"Elk: {elk_count} (Bulls: {elk_types['bulls']}, Cows: {elk_types['cows']})",
            f"Adaptive Conf: {adaptive_confidence:.3f} | Bull Thresh: {size_analysis['bull_threshold']:.0f}px",
            f"Movement: {'Yes' if movement_info['movement_detected'] else 'No'} | Mag: {movement_info['movement_magnitude']:.1f}"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(
                annotated_frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            cv2.putText(
                annotated_frame,
                line,
                (10, 25 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                1
            )
        
        # Save frame if elk detected or significant movement or every 15th processed frame
        should_save = (elk_count > 0 or 
                      movement_info['movement_detected'] or 
                      processed_frames % 15 == 0)
        
        if should_save:
            frame_filename = f"frame_{frame_idx:06d}_elk_{elk_count}_bulls_{elk_types['bulls']}_cows_{elk_types['cows']}_conf_{adaptive_confidence:.3f}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, annotated_frame)
            saved_frames += 1
            
            if elk_count > 0:
                print(f"Frame {frame_idx}: Found {elk_count} elk ({elk_types['bulls']} bulls, {elk_types['cows']} cows) "
                      f"- Conf: {adaptive_confidence:.3f}, Bull Thresh: {size_analysis['bull_threshold']:.0f}px")
        
        # Store comprehensive frame data
        frame_info = {
            'frame_idx': frame_idx,
            'timestamp': frame_idx / fps,
            'total_elk': elk_count,
            'bulls': elk_types['bulls'],
            'cows': elk_types['cows'],
            'adaptive_confidence': adaptive_confidence,
            'movement_info': movement_info,
            'size_analysis': size_analysis,
            'blur_score': elk_results['blur_score']
        }
        
        frame_data.append(frame_info)
        frame_idx += 1
        
        # Progress update
        if processed_frames % 50 == 0:
            print(f"Processed {processed_frames} frames, found {len(detection_results)} elk detections")
    
    cap.release()
    
    # Get final detection summary
    detection_summary = detector.get_detection_summary()
    
    # Analysis results
    print(f"\nDynamic Analysis Complete!")
    print(f"Processed {processed_frames} frames")
    print(f"Found {len(detection_results)} total elk detections")
    print(f"Saved {saved_frames} annotated frames")
    print(f"Detected {len(movement_events)} significant movement events")
    
    # Calculate comprehensive elk statistics
    if detection_results:
        # Frame-based statistics
        frames_with_elk = [f for f in frame_data if f['total_elk'] > 0]
        
        # Bull/cow statistics
        bull_detections = [d for d in detection_results if d['elk_type'] == 'bull']
        cow_detections = [d for d in detection_results if d['elk_type'] == 'cow']
        
        # Confidence statistics
        high_conf_detections = [d for d in detection_results if d['confidence'] > 0.5]
        
        # Size analysis
        all_sizes = [d['area'] for d in detection_results]
        size_range = (min(all_sizes), max(all_sizes)) if all_sizes else (0, 0)
        
        print(f"\nDynamic Elk Population Analysis:")
        print(f"  Maximum elk in single frame: {max_elk_in_frame}")
        print(f"  Maximum bulls in single frame: {max_bulls_in_frame}")
        print(f"  Maximum cows in single frame: {max_cows_in_frame}")
        print(f"  Frames with elk: {len(frames_with_elk)}/{processed_frames} ({len(frames_with_elk)/processed_frames*100:.1f}%)")
        print(f"  Size range detected: {size_range[0]:.0f} - {size_range[1]:.0f} pixels")
        
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
        print(f"  Movement events detected: {len(movement_events)}")
        
        # Dynamic elk count estimation using multiple methods
        estimated_elk_count = max_elk_in_frame  # Conservative: maximum seen in any frame
        estimated_bull_count = max_bulls_in_frame
        estimated_cow_count = max_cows_in_frame
        
        # Alternative estimation: most frequent high count
        elk_counts = [f['total_elk'] for f in frames_with_elk if f['total_elk'] > 0]
        if elk_counts:
            from collections import Counter
            count_frequency = Counter(elk_counts)
            most_frequent_count = count_frequency.most_common(1)[0][0]
            
            print(f"\n*** DYNAMIC ELK POPULATION ESTIMATE ***")
            print(f"Maximum observed method:")
            print(f"  Total Elk: {estimated_elk_count}")
            print(f"  Bulls: {estimated_bull_count}")
            print(f"  Cows (including calves): {estimated_cow_count}")
            print(f"Most frequent count method:")
            print(f"  Most frequent elk count: {most_frequent_count}")
            print(f"Bull/Cow Ratio: {estimated_bull_count/estimated_cow_count:.2f}" if estimated_cow_count > 0 else "Bull/Cow Ratio: N/A")
        
    else:
        print("No elk detected in the video.")
        estimated_elk_count = 0
        estimated_bull_count = 0
        estimated_cow_count = 0
        most_frequent_count = 0
    
    # Save comprehensive results
    summary_report = {
        'analysis_info': {
            'video_file': video_path,
            'analysis_date': datetime.now().isoformat(),
            'model_used': model_path,
            'min_elk_size': min_elk_size,
            'analysis_type': 'dynamic_adaptive_detection'
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
            'frames_with_elk': len([f for f in frame_data if f['total_elk'] > 0]),
            'movement_events': len(movement_events)
        },
        'elk_population_estimate': {
            'max_method': {
                'total_elk': estimated_elk_count,
                'bulls': estimated_bull_count,
                'cows_including_calves': estimated_cow_count,
                'bull_cow_ratio': estimated_bull_count/estimated_cow_count if estimated_cow_count > 0 else None
            },
            'frequent_method': {
                'most_frequent_count': most_frequent_count if 'most_frequent_count' in locals() else 0
            }
        },
        'detection_statistics': {
            'total_detections': len(detection_results),
            'bull_detections': len([d for d in detection_results if d['elk_type'] == 'bull']),
            'cow_detections': len([d for d in detection_results if d['elk_type'] == 'cow']),
            'high_confidence_detections': len([d for d in detection_results if d['confidence'] > 0.5]),
            'max_elk_single_frame': max_elk_in_frame,
            'max_bulls_single_frame': max_bulls_in_frame,
            'max_cows_single_frame': max_cows_in_frame,
            'size_range': size_range if 'size_range' in locals() else (0, 0)
        },
        'dynamic_analysis': {
            'size_progression': size_progression[:100],  # First 100 frames
            'confidence_progression': confidence_progression[:100],
            'movement_events': movement_events,
            'detection_summary': detection_summary
        },
        'detailed_detections': detection_results[:300]  # Save first 300 detections
    }
    
    # Save summary as JSON
    summary_path = os.path.join(output_dir, "dynamic_elk_analysis.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_report, f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  Annotated frames: {frames_dir}")
    
    return summary_report


def main():
    """Main function to run dynamic elk analysis."""
    # Configuration
    VIDEO_PATH = "data/raw/grassland.mp4"
    MODEL_PATH = "yolov8n.pt"
    OUTPUT_DIR = "results/dynamic_elk_analysis"
    MIN_ELK_SIZE = 50  # Ultra-low minimum size for distant elk
    
    # Check if files exist
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
        sys.exit(1)
    
    try:
        print("="*80)
        print("DYNAMIC ELK ANALYSIS - ADAPTIVE SIZE & MOVEMENT TRACKING")
        print("="*80)
        
        # Run dynamic analysis
        results = dynamic_elk_analysis(
            video_path=VIDEO_PATH,
            model_path=MODEL_PATH,
            output_dir=OUTPUT_DIR,
            min_elk_size=MIN_ELK_SIZE
        )
        
        print("\n" + "="*80)
        print("DYNAMIC ANALYSIS COMPLETE")
        print("="*80)
        print(f"Estimated Total Elk (Max Method): {results['elk_population_estimate']['max_method']['total_elk']}")
        print(f"Estimated Bulls: {results['elk_population_estimate']['max_method']['bulls']}")
        print(f"Estimated Cows (inc. calves): {results['elk_population_estimate']['max_method']['cows_including_calves']}")
        print(f"Most Frequent Count: {results['elk_population_estimate']['frequent_method']['most_frequent_count']}")
        print(f"Total Detections: {results['detection_statistics']['total_detections']}")
        print(f"Frames Processed: {results['processing_info']['frames_processed']}")
        print(f"Frames with Elk: {results['processing_info']['frames_with_elk']}")
        print(f"Movement Events: {results['processing_info']['movement_events']}")
        print(f"Size Range: {results['detection_statistics']['size_range'][0]:.0f} - {results['detection_statistics']['size_range'][1]:.0f} pixels")
        print("="*80)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
