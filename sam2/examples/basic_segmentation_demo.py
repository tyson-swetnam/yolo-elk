#!/usr/bin/env python3
"""
Basic SAM2 Segmentation Demo

This script demonstrates basic SAM2 segmentation capabilities
for elk detection in video footage.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import cv2
import numpy as np
from src.segmentation.sam2_segmenter import SAM2Segmenter


def main():
    """Run basic SAM2 segmentation demo."""
    
    # Configuration
    VIDEO_PATH = "data/raw/grassland.mp4"
    MODEL_CHECKPOINT = "sam2_hiera_large.pt"
    MODEL_CONFIG = "sam2_hiera_l.yaml"
    
    # Check if video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found: {VIDEO_PATH}")
        print("Please ensure the video file exists in the data/raw/ directory")
        return
    
    print("Initializing SAM2 Segmenter...")
    try:
        segmenter = SAM2Segmenter(
            model_checkpoint=MODEL_CHECKPOINT,
            model_config=MODEL_CONFIG,
            device="auto",
            confidence_threshold=0.5
        )
        print("SAM2 Segmenter initialized successfully!")
    except Exception as e:
        print(f"Warning: Could not initialize SAM2 models: {e}")
        print("Running in demo mode with dummy segmentation...")
        segmenter = None
    
    # Open video
    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")
    print("\nStarting Basic SAM2 Segmentation Demo...")
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  Q/ESC: Quit")
    print("  Click: Add point prompt for segmentation")
    
    # Create window
    cv2.namedWindow('SAM2 Basic Demo', cv2.WINDOW_RESIZABLE)
    
    # Mouse callback for point prompts
    click_points = []
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal click_points
        if event == cv2.EVENT_LBUTTONDOWN:
            click_points.append([x, y])
            print(f"Added point prompt at ({x}, {y})")
    
    cv2.setMouseCallback('SAM2 Basic Demo', mouse_callback)
    
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
            
            # Run segmentation if we have prompts and a working segmenter
            if segmenter and click_points:
                try:
                    prompts = {
                        'points': click_points,
                        'point_labels': [1] * len(click_points)  # All positive prompts
                    }
                    
                    results = segmenter.segment_frame(frame, prompts=prompts)
                    
                    # Visualize results
                    if len(results['masks']) > 0:
                        frame = segmenter.visualize_masks(
                            frame, 
                            results['masks'], 
                            results['scores'],
                            alpha=0.6
                        )
                        
                        # Show info
                        info_text = f"Frame: {frame_num}/{frame_count} | Masks: {len(results['masks'])}"
                        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                except Exception as e:
                    print(f"Segmentation error: {e}")
            
            elif not segmenter:
                # Demo mode - show dummy segmentation
                demo_text = "DEMO MODE - SAM2 not available"
                cv2.putText(frame, demo_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                info_text = f"Frame: {frame_num}/{frame_count} | Click to add prompts"
                cv2.putText(frame, info_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            else:
                # No prompts yet
                info_text = f"Frame: {frame_num}/{frame_count} | Click to add prompts"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw click points
            for i, point in enumerate(click_points):
                cv2.circle(frame, tuple(point), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i+1), (point[0]+10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display frame
            cv2.imshow('SAM2 Basic Demo', frame)
        
        # Handle key presses
        key = cv2.waitKey(int(frame_delay * 1000) if not paused else 30) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('c'):  # Clear prompts
            click_points = []
            print("Cleared all prompts")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if segmenter:
        segmenter.cleanup()
    
    print("Demo completed!")


if __name__ == "__main__":
    main()
