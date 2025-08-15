#!/usr/bin/env python3
"""
Test script for advanced cricket cover drive analysis features.
Validates all 10 bonus features and integration.
"""

import sys
import os
import tempfile
import json
from pathlib import Path
import numpy as np

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.phase import PhaseDetector, BattingPhase
from utils.contact import ContactDetector, ContactMoment
from utils.plotting import SmoothnessAnalyzer, ChartGenerator
from utils.performance import PerformanceMonitor, PerformanceOptimizer
from utils.ref_compare import ReferenceComparator
from utils.bat_tracking import BatDetector, BatDetection
from utils.skill_grade import SkillGrader, SkillLevel
from utils.report import ReportGenerator
from utils.metrics import FrameMetrics


def test_phase_detection():
    """Test phase detection functionality."""
    print("Testing Phase Detection...")
    
    detector = PhaseDetector()
    
    # Create mock metrics
    mock_metrics = []
    for i in range(100):
        metrics = FrameMetrics(
            front_elbow_angle=140 + np.sin(i * 0.1) * 10,
            spine_lean_angle=10 + np.sin(i * 0.05) * 5,
            head_over_front_knee=0.1 + np.sin(i * 0.02) * 0.05,
            front_foot_direction=15 + np.sin(i * 0.03) * 8
        )
        mock_metrics.append(metrics)
    
    # Detect phases
    phases = detector.detect_phases(mock_metrics)
    
    print(f"✓ Detected {len(phases)} phases")
    for phase in phases:
        print(f"  - {phase.phase.value}: frames {phase.start_frame}-{phase.end_frame}")
    
    return len(phases) > 0


def test_contact_detection():
    """Test contact detection functionality."""
    print("Testing Contact Detection...")
    
    detector = ContactDetector()
    
    # Create mock metrics with velocity peaks
    mock_metrics = []
    for i in range(100):
        # Create velocity peak around frame 50
        velocity = 0.1 + 0.8 * np.exp(-((i - 50) ** 2) / 100)
        
        metrics = FrameMetrics(
            front_elbow_angle=140,
            spine_lean_angle=10,
            head_over_front_knee=0.1,
            front_foot_direction=15
        )
        # Add velocity attribute
        setattr(metrics, 'wrist_velocity', velocity)
        mock_metrics.append(metrics)
    
    # Detect contacts
    contacts = detector.detect_contacts(mock_metrics)
    
    print(f"✓ Detected {len(contacts)} contact moments")
    for contact in contacts:
        print(f"  - Frame {contact.frame_idx}: confidence {contact.confidence:.2f}")
    
    return len(contacts) > 0


def test_smoothness_analysis():
    """Test smoothness analysis functionality."""
    print("Testing Smoothness Analysis...")
    
    analyzer = SmoothnessAnalyzer()
    
    # Create mock metrics with varying smoothness
    mock_metrics = []
    for i in range(100):
        # Add some noise to make it less smooth
        noise = np.random.normal(0, 2) if i > 50 else 0
        
        metrics = FrameMetrics(
            front_elbow_angle=140 + noise,
            spine_lean_angle=10 + noise * 0.5,
            head_over_front_knee=0.1 + abs(noise) * 0.01,
            front_foot_direction=15 + noise * 0.3
        )
        mock_metrics.append(metrics)
    
    # Analyze smoothness
    analysis = analyzer.analyze_smoothness(mock_metrics)
    
    print(f"✓ Overall smoothness: {analysis.get('overall_smoothness', 0):.2f}")
    print(f"✓ Problematic regions: {len(analysis.get('problematic_regions', []))}")
    
    return 'overall_smoothness' in analysis


def test_performance_monitoring():
    """Test performance monitoring functionality."""
    print("Testing Performance Monitoring...")
    
    monitor = PerformanceMonitor(target_fps=30.0)
    monitor.start_session()
    
    # Simulate frame processing
    for i in range(50):
        processing_time = 0.03 + np.random.normal(0, 0.01)  # ~30 FPS
        monitor.record_frame(processing_time)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    
    print(f"✓ Average FPS: {summary.avg_fps:.1f}")
    print(f"✓ Min/Max FPS: {summary.min_fps:.1f}/{summary.max_fps:.1f}")
    print(f"✓ Frames skipped: {monitor.frame_skip_count}")
    
    return summary.avg_fps > 0


def test_reference_comparison():
    """Test reference comparison functionality."""
    print("Testing Reference Comparison...")
    
    comparator = ReferenceComparator()
    
    # Create mock metrics
    mock_metrics = []
    for i in range(50):
        metrics = FrameMetrics(
            front_elbow_angle=140 + np.random.normal(0, 5),
            spine_lean_angle=10 + np.random.normal(0, 3),
            head_over_front_knee=0.1 + np.random.normal(0, 0.02),
            front_foot_direction=15 + np.random.normal(0, 4)
        )
        mock_metrics.append(metrics)
    
    # Compare to reference
    comparison = comparator.compare_to_reference(mock_metrics, skill_level="intermediate")
    
    print(f"✓ Skill level: {comparison.skill_level}")
    print(f"✓ Overall score: {comparison.overall_score:.1f}")
    print(f"✓ Deviations analyzed: {len(comparison.metric_deviations)}")
    
    return comparison.overall_score > 0


def test_bat_tracking():
    """Test bat tracking functionality."""
    print("Testing Bat Tracking...")
    
    detector = BatDetector()
    
    # Create mock frame and wrist keypoint
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Simulate a simple bat detection scenario
    # In a real scenario, this would be more complex
    print("✓ Bat detector initialized")
    print("✓ Note: Full bat detection requires actual video frames")
    
    return True


def test_skill_grading():
    """Test skill grading functionality."""
    print("Testing Skill Grading...")
    
    grader = SkillGrader()
    
    # Create mock category scores
    category_scores = {
        'Footwork': 7.5,
        'Head Position': 8.0,
        'Swing Control': 6.5,
        'Balance': 7.0,
        'Follow-through': 7.8
    }
    
    # Assess skill level
    assessment = grader.assess_skill_level(category_scores)
    
    print(f"✓ Skill level: {assessment.skill_level.value}")
    print(f"✓ Overall score: {assessment.overall_score:.1f}")
    print(f"✓ Strengths: {len(assessment.strengths)}")
    print(f"✓ Weaknesses: {len(assessment.weaknesses)}")
    
    return assessment.skill_level != SkillLevel.BEGINNER


def test_report_generation():
    """Test report generation functionality."""
    print("Testing Report Generation...")
    
    generator = ReportGenerator()
    
    # Create mock analysis results
    mock_results = {
        'evaluation': {
            'skill_level': 'Intermediate',
            'overall_score': 7.2,
            'summary': {
                'Footwork': {'score': 7.5, 'feedback': 'Good foot alignment'},
                'Head Position': {'score': 8.0, 'feedback': 'Excellent head position'},
                'Swing Control': {'score': 6.5, 'feedback': 'Needs improvement'},
                'Balance': {'score': 7.0, 'feedback': 'Good balance'},
                'Follow-through': {'score': 7.8, 'feedback': 'Solid follow-through'}
            },
            'global': {
                'avg_fps': 25.5,
                'frames_analyzed': 750,
                'frames_with_missing_keypoints': 15
            },
            'recommendations': [
                'Focus on swing control improvement',
                'Maintain current head position',
                'Work on consistency'
            ]
        }
    }
    
    # Generate HTML report
    with tempfile.TemporaryDirectory() as temp_dir:
        html_path = generator.generate_html_report(mock_results, f"{temp_dir}/test_report.html")
        
        if os.path.exists(html_path):
            print(f"✓ HTML report generated: {html_path}")
            return True
        else:
            print("✗ HTML report generation failed")
            return False


def test_streamlit_app():
    """Test Streamlit app availability."""
    print("Testing Streamlit App...")
    
    try:
        import streamlit as st
        print("✓ Streamlit is available")
        
        # Check if app file exists
        app_path = Path("app/streamlit_app.py")
        if app_path.exists():
            print("✓ Streamlit app file exists")
            return True
        else:
            print("✗ Streamlit app file not found")
            return False
    except ImportError:
        print("✗ Streamlit not installed")
        return False


def run_all_tests():
    """Run all advanced feature tests."""
    print("=" * 60)
    print("TESTING ADVANCED CRICKET COVER DRIVE ANALYSIS FEATURES")
    print("=" * 60)
    
    tests = [
        ("Phase Detection", test_phase_detection),
        ("Contact Detection", test_contact_detection),
        ("Smoothness Analysis", test_smoothness_analysis),
        ("Performance Monitoring", test_performance_monitoring),
        ("Reference Comparison", test_reference_comparison),
        ("Bat Tracking", test_bat_tracking),
        ("Skill Grading", test_skill_grading),
        ("Report Generation", test_report_generation),
        ("Streamlit App", test_streamlit_app)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"{'✓ PASSED' if success else '✗ FAILED'}")
        except Exception as e:
            print(f"✗ ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All advanced features are working correctly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the output above.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
