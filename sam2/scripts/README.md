# SAM2 Elk Analysis Scripts

This directory contains scripts for elk detection, segmentation, and tracking using SAM2 (Segment Anything Model 2).

## Scripts Overview

### Enhanced Elk Segmentation (`enhanced_elk_segmentation.py`)
The main script for real-time elk segmentation and tracking using SAM2.

**Features:**
- Real-time SAM2-based elk segmentation
- Elk classification (bulls, cows, calves) based on mask area and features
- Motion compensation for helicopter footage
- Interactive tracking with Norfair
- Comprehensive statistics and analysis
- Real-time visualization with information panels

**Usage:**
```bash
python enhanced_elk_segmentation.py [video_path] [config_path] [confidence_threshold]
```

**Controls:**
- `SPACE`: Pause/Resume
- `M`: Toggle Motion Compensation
- `F`: Toggle Feature Analysis
- `S`: Toggle Mask Display
- `T`: Toggle Tracking
- `Q/ESC`: Quit

**Example:**
```bash
# Use default grassland video
python enhanced_elk_segmentation.py

# Use specific video
python enhanced_elk_segmentation.py data/raw/burn.mp4

# Use custom config and confidence
python enhanced_elk_segmentation.py data/raw/grassland.mp4 configs/custom_config.yaml 0.6
```

## Configuration

Scripts use the configuration file at `configs/default_config.yaml`. Key settings include:

- **SAM2 Model**: Model checkpoint and configuration files
- **Segmentation**: Confidence thresholds, mask parameters
- **Tracking**: Norfair tracker settings optimized for masks
- **Motion Compensation**: Helicopter motion stabilization
- **Feature Analysis**: Elk-specific visual feature detection
- **Visualization**: Display options and colors

## Output

The scripts provide:

1. **Real-time Display**: Live segmentation and tracking visualization
2. **Statistics**: Elk counts, tracking metrics, video quality assessment
3. **Console Output**: Detailed analysis results and performance metrics
4. **Export Options**: Results can be saved via track manager

## Requirements

- SAM2 model files (download separately)
- Video files in `data/raw/` directory
- Python environment with SAM2 dependencies (see `environment.yml`)

## SAM2 vs YOLO Comparison

| Feature | SAM2 | YOLO |
|---------|------|------|
| Output | Pixel-level masks | Bounding boxes |
| Precision | Higher (exact boundaries) | Lower (rectangular bounds) |
| Occlusion Handling | Better separation | Limited overlap handling |
| Computational Cost | Higher | Lower |
| Interactive Prompting | Yes | No |
| Real-time Performance | Moderate | Fast |

## Troubleshooting

**SAM2 Model Not Found:**
- Download SAM2 model checkpoints
- Place in project root or update config paths
- Ensure model files match configuration

**Video Not Found:**
- Check video files exist in `data/raw/`
- Verify video format compatibility (MP4 recommended)
- Update video paths in configuration

**Performance Issues:**
- Reduce video resolution
- Lower confidence thresholds
- Disable motion compensation
- Use CPU instead of GPU if memory limited

## Advanced Usage

### Custom Prompting
Modify `elk_segmenter.py` to implement custom prompting strategies:
- Grid-based prompts
- Edge-detection prompts
- Adaptive content-based prompts

### Tracking Optimization
Adjust tracking parameters in configuration:
- Distance functions (IoU, Euclidean, mask IoU)
- Association thresholds
- Track initialization delays

### Feature Analysis
Customize elk feature detection:
- Color range adjustments
- Shape analysis parameters
- Size classification thresholds

## Integration with YOLO

The SAM2 system can be combined with YOLO for hybrid analysis:
1. Use YOLO for initial detection
2. Apply SAM2 for precise segmentation
3. Compare results for validation
4. Leverage strengths of both approaches
