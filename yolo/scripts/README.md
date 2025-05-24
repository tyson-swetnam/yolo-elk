# Elk Detection Scripts

This directory contains scripts for detecting and analyzing elk in helicopter footage using YOLO object detection with specialized enhancements for elk identification.

## Overview

The scripts are designed to work with the `grassland.mp4` video file containing elk footage taken from a helicopter. The system uses YOLO (You Only Look Once) object detection enhanced with elk-specific features including:

- **Elk Classification**: Distinguishes between bulls, cows, and calves based on size
- **Motion Compensation**: Handles helicopter movement and camera shake
- **Feature Analysis**: Analyzes elk-specific visual features (colors, white rump, aspect ratio)
- **Real-time Processing**: Provides smooth video playback with live detection

## Scripts

### 1. `enhanced_elk_detection.py` ⭐ **RECOMMENDED**

**Purpose**: Real-time elk detection with advanced features optimized for helicopter footage.

**Features**:
- Enhanced elk-specific detector with motion compensation
- Real-time classification of bulls, cows, and calves
- Interactive controls for toggling features
- Comprehensive statistics and visualization
- Elk-specific color analysis and feature detection

**Usage**:
```bash
conda activate yolo-tracking
python scripts/enhanced_elk_detection.py
```

**Controls**:
- `SPACE`: Pause/Resume playback
- `M`: Toggle motion compensation
- `F`: Toggle feature analysis
- `C`: Toggle confidence bars
- `Q/ESC`: Quit

**Output**: Real-time video display with elk detections, classification, and statistics

---

### 2. `elk_detection_with_output.py`

**Purpose**: Process video and save annotated frames with elk detections.

**Features**:
- Processes video frame-by-frame
- Saves annotated frames showing detections
- Generates detailed JSON summary report
- Focuses on elk-relevant YOLO classes only

**Usage**:
```bash
conda activate yolo-tracking
python scripts/elk_detection_with_output.py
```

**Output**:
- `results/elk_detection_output/annotated_frames/`: Saved frames with detections
- `results/elk_detection_output/elk_detection_summary.json`: Detailed analysis report

---

### 3. `test_elk_detector.py`

**Purpose**: Test and validate the ElkDetector functionality.

**Features**:
- Tests elk detection on sample frames
- Validates motion compensation and feature analysis
- Provides diagnostic output for troubleshooting

**Usage**:
```bash
conda activate yolo-tracking
python scripts/test_elk_detector.py
```

## Technical Details

### Elk Detection Strategy

The enhanced detection system uses the following approach:

1. **Target Classes**: Focuses on COCO classes that elk might be detected as:
   - Horse (class 17) - most similar to elk
   - Sheep (class 18) - for smaller elk/calves
   - Cow (class 19) - similar body mass
   - Bear (class 21) - for bulky bulls

2. **Size-Based Classification**:
   - **Calves**: Bounding box area < 1,500 pixels
   - **Cows**: Bounding box area 1,500 - 4,000 pixels
   - **Bulls**: Bounding box area > 4,000 pixels

3. **Motion Compensation**:
   - Uses ORB feature detection to track camera movement
   - Applies homography transformation to stabilize frames
   - Adjusts confidence thresholds for motion-blurred frames

4. **Elk Feature Analysis**:
   - **Color Analysis**: Looks for brown/tan elk colors in HSV space
   - **White Rump Detection**: Analyzes bottom third of detection for bright regions
   - **Aspect Ratio**: Prefers elongated detections (elk body shape)

### Color Coding

- **Green**: Bulls (largest elk)
- **Yellow**: Cows (medium elk)
- **Orange**: Calves (smallest elk)
- **Red**: Unknown/uncertain classifications

### Confidence Levels

- **Green Box**: High confidence (>0.7)
- **Yellow Box**: Medium confidence (0.5-0.7)
- **Orange Box**: Low confidence (<0.5)

## Requirements

### Environment Setup

1. **Activate Conda Environment**:
   ```bash
   conda activate yolo-tracking
   ```

2. **Required Files**:
   - `data/raw/grassland.mp4` - Input video file
   - `yolov8n.pt` - YOLO model weights

3. **Dependencies** (installed via `environment.yml`):
   - OpenCV (`opencv`)
   - Ultralytics YOLO (`ultralytics`)
   - NumPy, PyTorch, etc.

### Hardware Recommendations

- **GPU**: CUDA-compatible GPU recommended for real-time processing
- **RAM**: 8GB+ recommended
- **CPU**: Multi-core processor for video processing

## Output Examples

### Enhanced Detection Output

The enhanced detector provides:

```
Maximum elk counts observed:
  Bulls: 2
  Cows: 3
  Calves: 1
  Total: 6

Average elk per frame:
  Bulls: 0.8
  Cows: 1.2
  Calves: 0.3
  Total: 2.3

Video quality metrics:
  Frames with motion: 45/100 (45.0%)
  Average blur score: 156.7
```

### Detection Summary JSON

```json
{
  "analysis_info": {
    "video_file": "data/raw/grassland.mp4",
    "analysis_date": "2025-05-23T10:00:00",
    "model_used": "yolov8n.pt",
    "confidence_threshold": 0.25
  },
  "detection_results": {
    "total_detections": 45,
    "estimated_elk_count": 6,
    "max_elk_single_frame": 6,
    "frames_with_elk": 38,
    "class_distribution": {
      "horse": 15,
      "cow": 20,
      "sheep": 10
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**:
   - Ensure conda environment is activated: `conda activate yolo-tracking`

2. **Video file not found**:
   - Check that `data/raw/grassland.mp4` exists
   - Verify you're running from the project root directory

3. **YOLO model not found**:
   - Ensure `yolov8n.pt` is in the project root
   - Download if missing: The script will auto-download on first run

4. **Slow performance**:
   - Check if GPU is being used (look for CUDA messages)
   - Reduce video resolution or frame rate if needed
   - Close other applications to free up resources

5. **No detections found**:
   - Try lowering confidence threshold (edit script)
   - Check video quality and lighting conditions
   - Verify elk are visible and not too small/distant

### Performance Tips

1. **GPU Acceleration**: Ensure CUDA is properly installed for GPU acceleration
2. **Frame Skipping**: The detection scripts process every frame; consider skipping frames for faster processing
3. **Resolution**: Lower resolution videos process faster but may miss smaller elk
4. **Confidence Tuning**: Adjust confidence thresholds based on video quality

## Development Notes

### Current File Structure

```
scripts/
├── enhanced_elk_detection.py     # Main enhanced detector (RECOMMENDED)
├── elk_detection_with_output.py  # Batch processing with output
├── test_elk_detector.py          # Testing and validation
└── README.md                     # This file

src/detection/
├── yolo_detector.py              # Base YOLO detector
└── elk_detector.py               # Enhanced elk-specific detector

notebooks/
└── elk_detection_analysis.ipynb  # Analysis notebook

results/elk_detection_output/
├── elk_detection_summary.json    # Latest analysis results
└── annotated_frames/             # Saved detection frames
```

### Extending the System

To add new features:

1. **New Detection Classes**: Modify `elk_classes` in `ElkDetector`
2. **Custom Features**: Add methods to `analyze_elk_features()`
3. **Visualization**: Extend drawing methods in the player classes
4. **Export Formats**: Add new output formats in the processing scripts

### Contributing

When modifying the scripts:

1. Test with the provided `grassland.mp4` video
2. Ensure backward compatibility with existing features
3. Update this README with new features or changes
4. Follow the existing code style and documentation patterns

## License

This project is part of the YOLO-ELK tracking system. See the main project LICENSE file for details.
