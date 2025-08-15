# Cricket Cover Drive Analysis

A real-time biomechanical analysis system for cricket cover drive technique using computer vision and pose estimation. This project processes cricket videos to analyze batting technique and provide detailed feedback on five key aspects: Footwork, Head Position, Swing Control, Balance, and Follow-through.

## Features

- **Real-time Pose Estimation**: Uses MediaPipe BlazePose for accurate body keypoint detection
- **Biomechanical Metrics**: Calculates cricket-specific angles, distances, and velocities
- **Temporal Smoothing**: Applies EMA, Savitzky-Golay, and Kalman filtering for stable measurements
- **Live HUD Overlay**: Displays real-time metrics on video frames with feedback cues
- **Comprehensive Scoring**: Evaluates technique across five categories with detailed feedback
- **Robust Error Handling**: Gracefully handles missing keypoints and low-quality frames
- **Multiple Input Sources**: Supports local video files and YouTube URLs
- **CPU Optimized**: Designed to run efficiently on CPU without requiring GPU
- **Phase Segmentation**: Automatic detection of batting phases (stance, stride, downswing, impact, follow-through)
- **Contact Detection**: Identifies bat-ball contact moments
- **Performance Optimization**: Real-time processing with FPS monitoring
- **Reference Comparison**: Compares against ideal technique benchmarks
- **Skill Grading**: Provides Beginner/Intermediate/Advanced skill assessment
- **Comprehensive Reporting**: Generates HTML and PDF reports with detailed analysis
- **Streamlit Web App**: Interactive web interface for video analysis

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd AthleteRise
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install yt-dlp for YouTube support (optional but recommended):
```bash
pip install yt-dlp
```

## Usage

### Basic Usage

Analyze a local video file:
```bash
python cover_drive_analysis_realtime.py --source "path/to/video.mp4" --handedness right
```

Analyze a YouTube video:
```bash
python cover_drive_analysis_realtime.py --source "https://www.youtube.com/watch?v=VIDEO_ID" --handedness right
```

### Command Line Options

```bash
python cover_drive_analysis_realtime.py --source "<file-or-YouTube-URL>" --handedness right --resize_long_side 640 --target_fps 30
```

**Required Arguments:**
- `--source`: Video source (file path or YouTube URL)

**Optional Arguments:**
- `--handedness`: Batter handedness (`left` or `right`, default: `right`)
- `--resize_long_side`: Resize video long side to this value (default: `640`)
- `--target_fps`: Target FPS for processing (default: `30`)
- `--output_dir`: Output directory (default: `output`)
- `--test`: Run test harness mode

### Test Harness

Run with test harness to get performance metrics:
```bash
python cover_drive_analysis_realtime.py --source "test_video.mp4" --test
```

This will output:
- `avg_fps`: Average processing frames per second
- `frames_processed`: Total frames analyzed
- `missing_keypoint_counts`: Number of frames with missing keypoints

### Streamlit Web App

Launch the interactive web interface:
```bash
streamlit run app/streamlit_app.py
```

Features:
- Upload video files or YouTube URLs
- Real-time processing with progress tracking
- Interactive results display
- Download annotated videos and reports
- Side-by-side comparison views

## Output

The system generates comprehensive outputs:

### 1. Annotated Video (`output/annotated_video.mp4`)
- Original video with pose skeleton overlay
- Live metrics HUD showing:
  - Front elbow angle (degrees)
  - Spine lean angle (degrees)
  - Head-over-front-knee distance (normalized)
  - Front foot direction (degrees)
- Real-time feedback cues (✅ Good, ⚠️ Warning, ❌ Poor)
- Phase labels (stance, stride, downswing, impact, follow-through)
- Contact moment indicators
- Pose confidence indicators
- Frame information

### 2. Evaluation JSON (`output/evaluation.json`)
- Detailed scoring across five categories
- Skill grade assessment (Beginner/Intermediate/Advanced)
- Performance metrics and statistics
- Reference comparison deviations

### 3. Analysis Reports
- **HTML Report** (`output/report.html`): Interactive web report with charts and analysis
- **PDF Report** (`output/report.pdf`): Printable detailed analysis report
- **Smoothness Chart** (`output/metrics_smoothness.png`): Temporal consistency analysis

### 4. Debug Data
- **Landmark CSV** (`output/debug_landmarks.csv`): Per-frame landmark coordinates for analysis
```json
{
  "summary": {
    "Footwork": {"score": 7, "feedback": "Foot aligned; slight toe-out late in follow-through."},
    "Head Position": {"score": 8, "feedback": "Head remains over front knee during impact."},
    "Swing Control": {"score": 6, "feedback": "Elbow drops near impact; tighten arc."},
    "Balance": {"score": 8, "feedback": "Good stability and low lateral sway."},
    "Follow-through": {"score": 7, "feedback": "Follow-through is consistent; hold longer."}
  },
  "global": {
    "avg_fps": 12.4,
    "frames_analyzed": 742,
    "frames_with_missing_keypoints": 34,
    "handedness": "right"
  }
}
```

## Biomechanical Metrics

### Primary Metrics

1. **Front Elbow Angle**: Angle at elbow between vectors (shoulder → elbow) and (wrist → elbow)
2. **Spine Lean**: Angle between spine vector and vertical axis
3. **Head-over-Front-Knee**: Normalized horizontal distance between head and front knee
4. **Front Foot Direction**: Angle of foot axis relative to horizontal

### Additional Metrics

- Back elbow angle
- Knee separation
- Shoulder rotation
- Hip rotation
- Temporal velocities

## Scoring System

Each category is scored 1-10 based on:

### Footwork (7/10)
- **Foot Direction**: Alignment relative to target
- **Knee Separation**: Proper stance width

### Head Position (8/10)
- **Head-Knee Distance**: Maintaining position over front knee
- **Spine Lean**: Minimal forward lean

### Swing Control (6/10)
- **Elbow Angles**: Proper arm positioning
- **Elbow Velocity**: Smooth, controlled movement

### Balance (8/10)
- **Spine Stability**: Minimal lateral movement
- **Velocity Control**: Smooth motion

### Follow-through (7/10)
- **Elbow Maintenance**: Keeping good angles
- **Rotation**: Proper shoulder and hip rotation

## Technical Details

### Pose Estimation
- **Primary**: MediaPipe BlazePose (fast, CPU-optimized)
- **Fallback**: Optical flow propagation for missing frames
- **Temporal Refinement**: Kalman filtering and optical flow

### Smoothing Pipeline
1. **EMA**: Exponential Moving Average (α=0.25)
2. **Kalman**: State estimation for robust tracking
3. **Savitzky-Golay**: Polynomial smoothing (window=7, poly=2)

### Performance Targets
- **Angle MAE**: ≤6° (target: <5°)
- **Distance MAE**: ≤0.03 (normalized)
- **Processing Speed**: ≥6 FPS on modern CPU
- **Accuracy**: High-fidelity measurements with minimal jitter

## Error Handling

The system handles various error conditions:

- **Missing Keypoints**: Shows "—" in HUD, continues processing
- **Low Confidence**: Uses temporal propagation
- **Insufficient Detections**: Exits with error code 2 if <60% frames have pose
- **Video Issues**: Graceful handling of corrupted or unsupported formats

## Configuration

Modify `configs/thresholds.yml` to adjust:
- Scoring thresholds and weights
- Pose estimation parameters
- Smoothing coefficients
- Performance targets

## Development

### Project Structure
```
AthleteRise/
├── cover_drive_analysis_realtime.py  # Main CLI entrypoint
├── utils/
│   ├── pose.py                       # Pose estimation
│   ├── metrics.py                    # Biomechanical calculations
│   ├── smoothing.py                  # Temporal filtering
│   ├── overlay.py                    # Video overlay
│   ├── io.py                         # Video I/O
│   └── scoring.py                    # Final evaluation
├── configs/
│   └── thresholds.yml                # Configuration
├── requirements.txt                  # Dependencies
└── README.md                        # This file
```

### Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest --cov=utils tests/
```

### Code Quality

Format code:
```bash
black .
```

Lint code:
```bash
flake8 .
```

Type checking:
```bash
mypy .
```

## Performance Optimization

### For Better Accuracy
- Increase `resize_long_side` to 1280 or higher
- Use `model_complexity=2` in pose estimation
- Adjust smoothing parameters in `thresholds.yml`

### For Better Speed
- Decrease `resize_long_side` to 480 or lower
- Use `model_complexity=0` for lighter model
- Reduce smoothing window sizes

## Troubleshooting

### Common Issues

1. **"yt-dlp not found"**
   ```bash
   pip install yt-dlp
   ```

2. **"Could not open video source"**
   - Check file path is correct
   - Ensure video format is supported
   - Try different video codec

3. **"insufficient pose detections"**
   - Check video quality and lighting
   - Ensure batter is clearly visible
   - Try adjusting pose confidence thresholds

4. **Low FPS**
   - Reduce `resize_long_side`
   - Use lighter pose model
   - Close other applications

### Performance Tips

- Use SSD storage for faster video I/O
- Close unnecessary applications during processing
- Use shorter video clips for testing
- Monitor CPU usage and temperature

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe team for the excellent pose estimation library
- OpenCV community for computer vision tools
- Cricket coaching community for biomechanical insights

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review existing GitHub issues
3. Create a new issue with detailed information
4. Include video sample and error logs if applicable
