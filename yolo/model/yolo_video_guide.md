# YOLO Elk Detection Model

Elk model (elk_model.pt) is a WIP, more annotations needed. Accuracy differs based on environment 


## Basic Usage
```bash
python video_detector.py [INPUT_VIDEO] [OUTPUT_VIDEO] [OPTIONS]
```

**Required Arguments:**
- `INPUT_VIDEO`: Path to the input video file
- `OUTPUT_VIDEO`: Path where the processed video will be saved

## Configuration Options

### Model Selection (`--model`)

Choose different YOLO models based on your needs:

```bash
# Fastest, lowest accuracy
python video_detector.py input.mp4 output.mp4 --model elk_model.pt
```

### Confidence Threshold (`--conf`)

Control detection sensitivity:

```bash
# More sensitive (more detections, potentially more false positives)
python video_detector.py input.mp4 output.mp4 --conf 0.3

# Default balance
python video_detector.py input.mp4 output.mp4 --conf 0.5

# Less sensitive (fewer detections, higher confidence only)
python video_detector.py input.mp4 output.mp4 --conf 0.7
```

**Confidence Guidelines:**
- `0.3-0.4`: High sensitivity, may include uncertain detections
- `0.5`: Balanced (recommended for most cases)
- `0.6-0.8`: Conservative, only high-confidence detections

### IoU Threshold (`--iou`)

Control duplicate detection filtering:

```bash
# Less aggressive filtering (may keep overlapping boxes)
python video_detector.py input.mp4 output.mp4 --iou 0.3

# Default filtering
python video_detector.py input.mp4 output.mp4 --iou 0.45

# More aggressive filtering (removes more overlapping boxes)
python video_detector.py input.mp4 output.mp4 --iou 0.6
```

### Real-time Preview (`--preview`)

Watch processing in real-time:

```bash
python video_detector.py input.mp4 output.mp4 --preview
```

**Preview Controls:**
- Press `Q` to quit early
- Window shows processed frames as they're generated
- Useful for testing settings before full processing
