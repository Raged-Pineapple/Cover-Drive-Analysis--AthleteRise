#!/usr/bin/env python3
"""
Cricket Cover Drive Analysis - Real-time Processing
Processes cricket cover drive videos with pose estimation and biomechanical analysis.
"""
import shutil
import os

import argparse
import sys
import time
from pathlib import Path
import json
import csv
import logging
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
from tqdm import tqdm

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


class CoverDriveAnalyzer:
    """Main analyzer class that orchestrates the full pipeline."""

    def __init__(self, handedness: str = "right", resize_long_side: int = 640, target_fps: int = 30):
        self.handedness = handedness
        self.resize_long_side = resize_long_side
        self.target_fps = target_fps

        # Initialize components
        self.pose_estimator = PoseEstimator()
        self.metrics_calculator = MetricsCalculator(handedness=handedness)
        self.smoother = TemporalSmoother()
        self.overlay = VideoOverlay()
        self.video_io = VideoIO()
        self.score_calculator = ScoreCalculator()

        # Advanced analysis components
        self.phase_detector = PhaseDetector()
        self.contact_detector = ContactDetector()
        self.smoothness_analyzer = SmoothnessAnalyzer()
        self.chart_generator = ChartGenerator()
        self.performance_monitor = PerformanceMonitor(target_fps=target_fps)
        self.performance_optimizer = PerformanceOptimizer()
        self.reference_comparator = ReferenceComparator()
        self.bat_detector = BatDetector()
        self.skill_grader = SkillGrader()

        # Stats
        self.stats = {
            "frames_processed": 0,
            "frames_with_missing_keypoints": 0,
            "processing_times": [],
            "pose_confidences": []
        }

    def _open_writer_with_fallbacks(self, out_path: Path, fps: float, size: tuple) -> Optional[cv2.VideoWriter]:
        """Try multiple codecs to get a working writer."""
        # Preferred mp4v
        for fourcc_str in ("mp4v", "avc1", "H264", "XVID", "MJPG"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(out_path), fourcc, max(fps, 1.0), size)
            if writer.isOpened():
                logging.info(f"Opened video writer with codec: {fourcc_str}")
                return writer
        logging.error("Failed to open video writer with any available codec.")
        return None

    def process_video(self, video_path: str, handedness: str, skill_level: str, 
                      enable_bat_tracking: bool, enable_phase_detection: bool, 
                      progress_callback=None) -> Dict[str, Any]:
        """Process video, run analysis, and return results with annotated video path."""
        
        logging.info(f"--- Starting video processing for {video_path} ---")
        logging.info(f"Parameters: handedness={handedness}, skill_level={skill_level}, bat_tracking={enable_bat_tracking}, phase_detection={enable_phase_detection}")

        self.performance_monitor.start_session()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Could not open video file.")
            return {"error": "Could not open video file."}

        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Setup video writer
        output_dir = os.path.join(os.path.dirname(video_path), "analysis_results")
        os.makedirs(output_dir, exist_ok=True)
        output_video_path = os.path.join(output_dir, f"annotated_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # More compatible codec
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # --- Initialize all analysis components ---
        pose_estimator = PoseEstimator()
        metrics_calculator = MetricsCalculator(handedness=handedness)
        bat_detector = BatDetector() if enable_bat_tracking else None
        phase_detector = PhaseDetector() if enable_phase_detection else None
        
        # --- Data storage ---
        metrics_history: List[FrameMetrics] = []
        bat_detections: List[BatDetection] = []
        frame_count = 0

        # --- Main processing loop ---
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            timestamp = frame_count / fps
            annotated_frame = frame.copy()

            # 1. Pose Estimation
            pose_results = pose_estimator.estimate_pose(frame)
            
            if pose_results and 'landmarks' in pose_results:
                landmarks = pose_results['landmarks']
                
                # 2. Metrics Calculation
                frame_metrics = metrics_calculator.calculate_frame_metrics(landmarks, frame.shape, timestamp)
                metrics_history.append(frame_metrics)
                
                # 3. Bat Tracking (if enabled)
                if bat_detector:
                    wrist_kp_name = pose_estimator.get_handedness_keypoints(landmarks, handedness)['front_wrist']
                    wrist_kp = landmarks.get(wrist_kp_name)
                    if wrist_kp and wrist_kp.confidence > 0.3:
                        bat_detection = bat_detector.detect_bat(frame, wrist_kp, frame_count)
                        if bat_detection:
                            bat_detections.append(bat_detection)
                            annotated_frame = self.overlay_drawer.draw_bat(annotated_frame, bat_detection)

                # Draw pose overlay
                annotated_frame = self.overlay_drawer.draw_pose(annotated_frame, landmarks, pose_results['confidence'])
            
            # Write frame to output video
            out.write(annotated_frame)
            
            # Performance tracking and progress update
            processing_time = time.time() - frame_start_time
            self.performance_monitor.record_frame(processing_time)
            if progress_callback:
                progress = (frame_count + 1) / total_frames
                progress_callback(progress)
            
            frame_count += 1

        cap.release()
        out.release()
        
        # --- Post-processing and Reporting ---
        if not metrics_history:
            logging.error("No pose detected, cannot perform analysis.")
            return {"error": "No pose detected, cannot perform analysis."}

        logging.info(f"--- Frame processing complete. Total frames: {len(metrics_history)} ---")

        # 4. Phase Detection
        phase_segments = phase_detector.detect_phases(metrics_history) if phase_detector else []
        phase_summary = phase_detector.get_phase_summary(phase_segments) if phase_detector else {}

        # 5. Swing Analysis
        swing_analysis = bat_detector.get_swing_analysis() if bat_detector else {}

        # 6. Scoring
        score_calculator = ScoreCalculator(skill_level=skill_level)
        evaluation = score_calculator.calculate_final_scores(
            metrics_history=metrics_history, 
            phases=phase_segments, 
            swing_analysis=swing_analysis
        )

        logging.info(f"--- Scoring complete. Score: {evaluation['final_score']} ---")
        logging.debug(f"Full scoring results: {json.dumps(evaluation, indent=2)}")

        # 7. Report Generation
        report_generator = ReportGenerator(evaluation, metrics_history, phase_segments, swing_analysis)
        report_path_html = os.path.join(output_dir, "report.html")
        report_generator.generate_html_report(report_path_html)
        
        # (Optional) PDF Report
        # report_path_pdf = os.path.join(output_dir, "report.pdf")
        # report_generator.generate_pdf_report(report_path_pdf)

        performance_summary = self.performance_monitor.get_performance_summary()

        logging.info(f"--- Process video finished. Returning results. ---")
        logging.debug(f"Final results dictionary: {json.dumps({'annotated_video_path': output_video_path, 'evaluation': evaluation, 'report_path': report_path_html, 'performance': performance_summary, 'metrics_history': [m.__dict__ for m in metrics_history], 'phases': phase_summary}, indent=2)}")

        return {
            "annotated_video_path": output_video_path,
            "evaluation": evaluation,
            "report_path": report_path_html,
            "performance": performance_summary,
            "metrics_history": [m.__dict__ for m in metrics_history],
            "phases": phase_summary
        }

    def process_video_original(self, source: str, output_dir: str = "output", enable_bat_tracking: bool = True, enable_phase_detection: bool = True, skill_level: str = 'intermediate') -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cap = self.video_io.get_video_capture(source, self.resize_long_side)
        if not cap or not cap.isOpened():
            logging.error(f"Could not open video source: {source}")
            raise ValueError(f"Could not open video source: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_video_path = output_path / f"{Path(source).stem}_annotated.mp4"
        writer = self._open_writer_with_fallbacks(output_video_path, fps, (width, height))
        if not writer:
            raise RuntimeError("Failed to open VideoWriter with any available codec.")

        print(f"[INFO] Processing video: {width}x{height} @ {fps:.1f}fps. Output: {output_video_path}")

        all_landmarks, all_metrics, all_bat_tips = [], [], []

        self.performance_monitor.start_session()
        for frame_idx in tqdm(range(total_frames), desc="Analyzing Frames"):
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()

            optimized_frame = self.performance_optimizer.optimize_frame(frame)
            pose_result = self.pose_estimator.estimate_pose(optimized_frame)

            if pose_result and pose_result.get("landmarks"):
                self.stats["pose_confidences"].append(pose_result.get('confidence', 0))
                metrics = self.metrics_calculator.calculate_frame_metrics(pose_result["landmarks"], frame.shape[:2])
            else:
                self.stats["frames_with_missing_keypoints"] += 1
                metrics = self.metrics_calculator.get_empty_metrics()

            all_landmarks.append(pose_result.get("landmarks", []))
            all_metrics.append(metrics)

            if enable_bat_tracking:
                bat_tip = self.bat_detector.detect_bat_tip(frame, pose_result.get("landmarks"))
                all_bat_tips.append(bat_tip)
                pose_result['bat_tip'] = bat_tip

            annotated_frame = self.overlay.draw_overlay(frame, pose_result, metrics, frame_idx)
            writer.write(annotated_frame)
            self.stats["frames_processed"] += 1
            
            processing_time = time.time() - frame_start_time
            self.performance_monitor.record_frame(processing_time)

        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if not output_video_path.exists() or output_video_path.stat().st_size < 1024:
            raise RuntimeError(f"Annotated video was not created or is empty: {output_video_path}")

        print(f"[INFO] Finished writing annotated video: {output_video_path}")

        # Post-processing and evaluation
        smoothed_metrics = self.smoother.smooth_all_metrics(all_metrics)
        phase_analysis = self.phase_detector.detect_phases(smoothed_metrics) if enable_phase_detection else {}
        contact_analysis = self.contact_detector.detect_contacts(smoothed_metrics, all_bat_tips)
        smoothness_analysis = self.smoothness_analyzer.analyze_smoothness(smoothed_metrics, phase_analysis.get('phases', []))
        
        reference_comparison = self.reference_comparator.compare_to_reference(
            all_landmarks, smoothed_metrics, skill_level=skill_level
        )

        evaluation = self.score_calculator.calculate_scores(
            smoothed_metrics, phase_analysis, contact_analysis, reference_comparison, smoothness_analysis
        )

        skill_grade_result = self.skill_grader.grade_skill(evaluation)
        evaluation.update(skill_grade_result)

        # Generate charts and save paths
        chart_paths = self.chart_generator.generate_all_charts(smoothed_metrics, phase_analysis, output_path)

        # Final results package
        results = {
            "evaluation": evaluation,
            "annotated_video_path": str(output_video_path),
            "chart_paths": chart_paths,
            "phase_analysis": phase_analysis,
            "contact_analysis": contact_analysis,
            "smoothness_analysis": smoothness_analysis,
            "reference_comparison": reference_comparison,
            "landmarks_data": all_landmarks,
            "metrics_data": smoothed_metrics,
            "global": {
                "avg_fps": self.performance_monitor.get_performance_summary().avg_fps,
                "total_processing_time": sum(self.performance_monitor.get_performance_summary().frame_processing_times),
                "frames_analyzed": self.stats["frames_processed"],
                "frames_with_missing_keypoints": self.stats["frames_with_missing_keypoints"],
            }
        }

        # Generate comprehensive HTML report
        try:
            report_path = generate_comprehensive_report(results, output_dir)
            results["report_paths"] = {"html": str(report_path)}
        except Exception as e:
            print(f"[WARN] Could not generate HTML report: {e}")

        # Save full results to JSON for debugging
        with open(output_path / "full_analysis_results.json", "w") as f:
            json.dump(results, f, indent=4, default=lambda o: str(o) if isinstance(o, (np.ndarray, Path)) else o)

        return results



def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Cricket Cover Drive Analysis - Real-time Processing")
    parser.add_argument("--source", required=True, help="Video source (file path or YouTube URL)")
    parser.add_argument("--handedness", choices=["left", "right"], default="right", help="Batter handedness")
    parser.add_argument("--resize_long_side", type=int, default=640, help="Resize long side (px)")
    parser.add_argument("--target_fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--output_dir", default="output", help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run test harness")
    args = parser.parse_args()

    try:
        analyzer = CoverDriveAnalyzer(
            handedness=args.handedness,
            resize_long_side=args.resize_long_side,
            target_fps=args.target_fps,
        )
        result = analyzer.process_video(args.source, args.output_dir)

        if args.test:
            print("\n=== TEST HARNESS RESULTS ===")
            if "error" in result:
                print(f"ERROR: {result['error']}")
                sys.exit(2)
            else:
                print(f"avg_fps: {result['global']['avg_fps']:.2f}")
                print(f"frames_processed: {result['global']['frames_analyzed']}")
                print(f"missing_keypoint_counts: {result['global']['frames_with_missing_keypoints']}")

        sys.exit(0 if "error" not in result else 2)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
