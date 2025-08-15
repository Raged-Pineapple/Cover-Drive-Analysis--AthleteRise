#!/usr/bin/env python3
"""
Test harness for Cricket Cover Drive Analysis.
Validates system components and provides performance metrics.
"""

import sys
import time
import json
from pathlib import Path
import argparse
import numpy as np

from utils.pose import PoseEstimator
from utils.metrics import MetricsCalculator
from utils.smoothing import TemporalSmoother
from utils.overlay import VideoOverlay
from utils.io import VideoIO
from utils.scoring import ScoreCalculator
from utils.phase import PhaseDetector
from utils.contact import ContactDetector
from utils.plotting import SmoothnessAnalyzer, ChartGenerator
from utils.performance import PerformanceMonitor, PerformanceOptimizer
from utils.ref_compare import ReferenceComparator
from utils.bat_tracking import BatDetector
from utils.skill_grade import SkillGrader
from utils.report import generate_comprehensive_report


def test_pose_estimation():
    """Test pose estimation with synthetic data."""
    print("Testing pose estimation...")
    
    try:
        pose_estimator = PoseEstimator()
        print("✓ PoseEstimator initialized successfully")
        
        # Test keypoint mapping
        handedness_map = pose_estimator.get_handedness_keypoints({}, "right")
        assert 'front_elbow' in handedness_map
        print("✓ Handedness mapping working")
        
        return True
    except Exception as e:
        print(f"✗ Pose estimation test failed: {e}")
        return False


def test_metrics_calculation():
    """Test metrics calculation with synthetic data."""
    print("Testing metrics calculation...")
    
    try:
        metrics_calc = MetricsCalculator(handedness="right")
        print("✓ MetricsCalculator initialized successfully")
        
        # Test empty metrics
        empty_metrics = metrics_calc.get_empty_metrics()
        assert empty_metrics is not None
        print("✓ Empty metrics generation working")
        
        # Test metric names
        metric_names = metrics_calc.get_metric_names()
        assert len(metric_names) > 0
        print("✓ Metric names retrieval working")
        
        return True
    except Exception as e:
        print(f"✗ Metrics calculation test failed: {e}")
        return False


def test_smoothing():
    """Test temporal smoothing components."""
    print("Testing temporal smoothing...")
    
    try:
        smoother = TemporalSmoother()
        print("✓ TemporalSmoother initialized successfully")
        
        # Test reset functionality
        smoother.reset()
        print("✓ Smoothing reset working")
        
        return True
    except Exception as e:
        print(f"✗ Smoothing test failed: {e}")
        return False


def test_video_io():
    """Test video I/O components."""
    print("Testing video I/O...")
    
    try:
        video_io = VideoIO()
        print("✓ VideoIO initialized successfully")
        
        # Test URL validation
        is_youtube = video_io._is_youtube_url("https://www.youtube.com/watch?v=test")
        assert is_youtube
        print("✓ YouTube URL detection working")
        
        return True
    except Exception as e:
        print(f"✗ Video I/O test failed: {e}")
        return False


def test_scoring():
    """Test scoring system."""
    print("Testing scoring system...")
    
    try:
        score_calc = ScoreCalculator()
        print("✓ ScoreCalculator initialized successfully")
        
        # Test error result creation
        error_result = score_calc._create_error_result("Test error")
        assert "error" in error_result
        print("✓ Error result creation working")
        
        return True
    except Exception as e:
        print(f"✗ Scoring test failed: {e}")
        return False


def test_overlay():
    """Test video overlay components."""
    print("Testing video overlay...")
    
    try:
        overlay = VideoOverlay()
        print("✓ VideoOverlay initialized successfully")
        
        # Test color scheme
        assert 'skeleton' in overlay.colors
        print("✓ Color scheme working")
        
        return True
    except Exception as e:
        print(f"✗ Overlay test failed: {e}")
        return False


def test_phase_detection():
    """Test phase detection functionality."""
    print("Testing phase detection...")
    
    try:
        phase_detector = PhaseDetector()
        print("✓ PhaseDetector initialized successfully")
        
        # Create mock metrics data
        mock_metrics = []
        for i in range(100):
            # Simulate different phases
            if i < 20:  # stance
                metrics = type('MockMetrics', (), {
                    'front_elbow_angle': 90 + np.random.normal(0, 5),
                    'spine_lean_angle': 5 + np.random.normal(0, 2),
                    'wrist_speed': 0.1 + np.random.normal(0, 0.05)
                })()
            elif i < 40:  # stride
                metrics = type('MockMetrics', (), {
                    'front_elbow_angle': 85 + np.random.normal(0, 5),
                    'spine_lean_angle': 8 + np.random.normal(0, 2),
                    'wrist_speed': 0.3 + np.random.normal(0, 0.1)
                })()
            elif i < 60:  # downswing
                metrics = type('MockMetrics', (), {
                    'front_elbow_angle': 70 + np.random.normal(0, 5),
                    'spine_lean_angle': 15 + np.random.normal(0, 3),
                    'wrist_speed': 0.8 + np.random.normal(0, 0.2)
                })()
            elif i < 80:  # impact
                metrics = type('MockMetrics', (), {
                    'front_elbow_angle': 60 + np.random.normal(0, 5),
                    'spine_lean_angle': 20 + np.random.normal(0, 3),
                    'wrist_speed': 1.2 + np.random.normal(0, 0.3)
                })()
            else:  # follow-through
                metrics = type('MockMetrics', (), {
                    'front_elbow_angle': 75 + np.random.normal(0, 5),
                    'spine_lean_angle': 12 + np.random.normal(0, 3),
                    'wrist_speed': 0.6 + np.random.normal(0, 0.2)
                })()
            mock_metrics.append(metrics)
        
        # Test phase detection
        phases = phase_detector.detect_phases(mock_metrics)
        assert len(phases) > 0
        print("✓ Phase detection working")
        
        return True
    except Exception as e:
        print(f"✗ Phase detection test failed: {e}")
        return False


def test_contact_detection():
    """Test contact detection functionality."""
    print("Testing contact detection...")
    
    try:
        contact_detector = ContactDetector()
        print("✓ ContactDetector initialized successfully")
        
        # Create mock metrics with velocity peaks
        mock_metrics = []
        for i in range(100):
            # Create a velocity peak around frame 50
            if 45 <= i <= 55:
                wrist_velocity = 2.0 + np.random.normal(0, 0.5)
            else:
                wrist_velocity = 0.1 + np.random.normal(0, 0.05)
            
            metrics = type('MockMetrics', (), {
                'wrist_velocity': wrist_velocity,
                'front_elbow_angle': 70 + np.random.normal(0, 5),
                'spine_lean_angle': 15 + np.random.normal(0, 3)
            })()
            mock_metrics.append(metrics)
        
        # Test contact detection
        contacts = contact_detector.detect_contacts(mock_metrics, 30.0)
        print("✓ Contact detection working")
        
        return True
    except Exception as e:
        print(f"✗ Contact detection test failed: {e}")
        return False


def test_smoothness_analysis():
    """Test smoothness analysis functionality."""
    print("Testing smoothness analysis...")
    
    try:
        smoothness_analyzer = SmoothnessAnalyzer()
        print("✓ SmoothnessAnalyzer initialized successfully")
        
        # Create mock metrics with varying smoothness
        mock_metrics = []
        for i in range(100):
            # Add some noise to simulate jitter
            noise_level = 0.1 if i < 50 else 0.5  # More noise in second half
            metrics = type('MockMetrics', (), {
                'front_elbow_angle': 90 + np.random.normal(0, noise_level * 10),
                'spine_lean_angle': 10 + np.random.normal(0, noise_level * 5),
                'head_over_front_knee': 0.5 + np.random.normal(0, noise_level * 0.1),
                'front_foot_direction': 15 + np.random.normal(0, noise_level * 3)
            })()
            mock_metrics.append(metrics)
        
        # Test smoothness analysis
        analysis = smoothness_analyzer.analyze_smoothness(mock_metrics)
        assert 'overall_smoothness' in analysis
        print("✓ Smoothness analysis working")
        
        return True
    except Exception as e:
        print(f"✗ Smoothness analysis test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("Testing performance monitoring...")
    
    try:
        performance_monitor = PerformanceMonitor(target_fps=30)
        print("✓ PerformanceMonitor initialized successfully")
        
        # Test performance tracking
        performance_monitor.start_session()
        for i in range(10):
            performance_monitor.record_frame(0.033)  # 30 FPS
        
        summary = performance_monitor.get_performance_summary()
        assert summary.avg_fps > 0
        print("✓ Performance monitoring working")
        
        return True
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        return False


def test_reference_comparison():
    """Test reference comparison functionality."""
    print("Testing reference comparison...")
    
    try:
        reference_comparator = ReferenceComparator()
        print("✓ ReferenceComparator initialized successfully")
        
        # Create mock metrics
        mock_metrics = []
        for i in range(50):
            metrics = type('MockMetrics', (), {
                'front_elbow_angle': 85 + np.random.normal(0, 5),
                'spine_lean_angle': 12 + np.random.normal(0, 3),
                'head_over_front_knee': 0.6 + np.random.normal(0, 0.1),
                'front_foot_direction': 18 + np.random.normal(0, 2)
            })()
            mock_metrics.append(metrics)
        
        # Test reference comparison
        comparison = reference_comparator.compare_to_reference(mock_metrics, {}, {})
        assert hasattr(comparison, 'skill_level')
        print("✓ Reference comparison working")
        
        return True
    except Exception as e:
        print(f"✗ Reference comparison test failed: {e}")
        return False


def test_bat_detection():
    """Test bat detection functionality."""
    print("Testing bat detection...")
    
    try:
        bat_detector = BatDetector()
        print("✓ BatDetector initialized successfully")
        
        # Create mock frame and wrist keypoint
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_wrist = type('MockKeyPoint', (), {'x': 320, 'y': 240, 'confidence': 0.9})()
        
        # Test bat detection
        detection = bat_detector.detect_bat(mock_frame, mock_wrist, 0)
        print("✓ Bat detection working")
        
        return True
    except Exception as e:
        print(f"✗ Bat detection test failed: {e}")
        return False


def test_skill_grading():
    """Test skill grading functionality."""
    print("Testing skill grading...")
    
    try:
        skill_grader = SkillGrader()
        print("✓ SkillGrader initialized successfully")
        
        # Create mock evaluation data
        mock_summary = {
            'footwork': 7.5,
            'head_position': 8.0,
            'swing_control': 6.5,
            'balance': 7.0,
            'follow_through': 8.5
        }
        
        mock_smoothness = {'overall_smoothness': 0.75}
        mock_reference = type('MockReference', (), {'skill_level': 'intermediate'})()
        
        # Test skill grading
        assessment = skill_grader.assess_skill_level(mock_summary, mock_smoothness, mock_reference)
        assert hasattr(assessment, 'skill_level')
        print("✓ Skill grading working")
        
        return True
    except Exception as e:
        print(f"✗ Skill grading test failed: {e}")
        return False


def test_report_generation():
    """Test report generation functionality."""
    print("Testing report generation...")
    
    try:
        # Create mock evaluation data
        mock_evaluation = {
            'summary': {
                'footwork': 7.5,
                'head_position': 8.0,
                'swing_control': 6.5,
                'balance': 7.0,
                'follow_through': 8.5
            },
            'skill_level': 'intermediate',
            'overall_score': 7.5,
            'global': {
                'avg_fps': 25.0,
                'frames_analyzed': 1000,
                'frames_with_missing_keypoints': 50
            }
        }
        
        # Test report generation
        output_dir = "test_output"
        Path(output_dir).mkdir(exist_ok=True)
        
        report_paths = generate_comprehensive_report(mock_evaluation, output_dir)
        assert isinstance(report_paths, dict)
        print("✓ Report generation working")
        
        return True
    except Exception as e:
        print(f"✗ Report generation test failed: {e}")
        return False


def run_component_tests():
    """Run all component tests."""
    print("=== Component Tests ===\n")
    
    tests = [
        test_pose_estimation,
        test_metrics_calculation,
        test_smoothing,
        test_video_io,
        test_scoring,
        test_overlay,
        test_phase_detection,
        test_contact_detection,
        test_smoothness_analysis,
        test_performance_monitoring,
        test_reference_comparison,
        test_bat_detection,
        test_skill_grading,
        test_report_generation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    return passed == total


def validate_video_processing(video_path: str):
    """Validate video processing pipeline."""
    print(f"=== Video Processing Validation ===")
    print(f"Video: {video_path}")
    
    try:
        # Initialize components
        video_io = VideoIO()
        pose_estimator = PoseEstimator()
        metrics_calc = MetricsCalculator()
        smoother = TemporalSmoother()
        
        # Get video info
        video_info = video_io.get_video_info(video_path)
        if not video_info:
            print("✗ Could not get video information")
            return False
        
        print(f"✓ Video info: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps")
        
        # Test video capture
        cap = video_io.get_video_capture(video_path)
        if not cap:
            print("✗ Could not open video capture")
            return False
        
        print("✓ Video capture working")
        
        # Process a few frames
        frame_count = 0
        pose_count = 0
        start_time = time.time()
        
        for _ in range(10):  # Process first 10 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            # Pose estimation
            pose_result = pose_estimator.estimate_pose(frame)
            if pose_result:
                pose_count += 1
                
                # Metrics calculation
                metrics = metrics_calc.calculate_frame_metrics(
                    pose_result['landmarks'], frame.shape[:2]
                )
                
                # Smoothing
                smoothed_metrics = smoother.smooth_metrics(metrics)
            
            frame_count += 1
        
        processing_time = time.time() - start_time
        
        cap.release()
        
        # Calculate metrics
        fps = frame_count / processing_time if processing_time > 0 else 0
        pose_detection_rate = pose_count / frame_count if frame_count > 0 else 0
        
        print(f"✓ Processed {frame_count} frames in {processing_time:.2f}s")
        print(f"✓ Average FPS: {fps:.1f}")
        print(f"✓ Pose detection rate: {pose_detection_rate:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Video processing validation failed: {e}")
        return False


def create_test_video():
    """Create a simple test video for validation."""
    print("=== Creating Test Video ===")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple test video
        width, height = 640, 480
        fps = 30
        duration = 3  # seconds
        
        output_path = "test_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame_idx in range(fps * duration):
            # Create a frame with moving shapes to simulate motion
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Add moving rectangle
            x = int((frame_idx / (fps * duration)) * width)
            cv2.rectangle(frame, (x, 200), (x + 100, 300), (0, 255, 0), -1)
            
            # Add static elements
            cv2.circle(frame, (320, 240), 50, (255, 0, 0), -1)
            
            out.write(frame)
        
        out.release()
        print(f"✓ Test video created: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"✗ Test video creation failed: {e}")
        return None


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test harness for Cricket Cover Drive Analysis")
    parser.add_argument("--video", help="Path to test video file")
    parser.add_argument("--create-test-video", action="store_true", help="Create a test video")
    parser.add_argument("--components-only", action="store_true", help="Run only component tests")
    
    args = parser.parse_args()
    
    print("Cricket Cover Drive Analysis - Test Harness")
    print("=" * 50)
    
    # Run component tests
    component_success = run_component_tests()
    
    if args.components_only:
        sys.exit(0 if component_success else 1)
    
    # Video processing validation
    video_path = args.video
    if not video_path:
        if args.create_test_video:
            video_path = create_test_video()
        else:
            # Look for existing test video
            test_videos = ["test_video.mp4", "sample_video.mp4", "cricket_video.mp4"]
            for test_vid in test_videos:
                if Path(test_vid).exists():
                    video_path = test_vid
                    break
    
    if video_path and Path(video_path).exists():
        video_success = validate_video_processing(video_path)
        print(f"\nOverall Result: {'PASS' if component_success and video_success else 'FAIL'}")
        sys.exit(0 if component_success and video_success else 1)
    else:
        print(f"\nNo test video available. Use --create-test-video or provide --video path")
        print(f"Component tests: {'PASS' if component_success else 'FAIL'}")
        sys.exit(0 if component_success else 1)


if __name__ == "__main__":
    main()
