"""
Streamlit web application for Cricket Cover Drive Analysis.
Provides interactive interface for video upload, processing, and results display.
"""

import streamlit as st
import sys
import shutil
from pathlib import Path
import json
from typing import Optional, Dict, Any
import yt_dlp
import cv2
from dotenv import load_dotenv
from os import getenv

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, parent_dir)

# Import from utils
from utils.gemini_analysis import GeminiAnalyzer
from utils.plotting import ChartGenerator
from utils.skill_grade import SkillGrader
from utils.report import ReportGenerator
from utils.metrics import MetricsCalculator
from utils.pose import PoseEstimator
from utils.ref_compare import ReferenceComparator

import importlib.util

# --- BEGIN INLINED ANALYSIS CLASS ---
# This class is inlined to prevent issues with cached Python modules.

class CoverDriveAnalyzer:
    def _compute_category_scores(self, metrics_history: list, skill_level: str) -> dict:
        """Placeholder to compute category scores from metrics."""
        # This is a simplified placeholder. A real implementation would involve
        # more complex logic to derive scores from the metrics_history.
        if not metrics_history:
            return {}

        # Example: Calculate average of a metric
        valid_angles = [m.front_elbow_angle for m in metrics_history if m.front_elbow_angle is not None]
        avg_front_elbow_angle = sum(valid_angles) / len(valid_angles) if valid_angles else 140  # Default angle if none found

        # Simple scoring logic (placeholders)
        scores = {
            'Footwork': 7.0 + (1 if skill_level == 'advanced' else 0),
            'Head Position': 8.0,
            # Score swing control based on the average front elbow angle (placeholder logic)
            'Swing Control': 5.0 + (avg_front_elbow_angle / 180.0) * 5.0,
            'Balance': 7.5,
            'Follow-through': 7.0
        }
        
        # Ensure scores are within a 0-10 range
        for key in scores:
            scores[key] = max(0.0, min(10.0, scores[key]))
            
        return scores
    """Main analyzer class that orchestrates the full pipeline."""

    def __init__(self, handedness: str = "right", resize_long_side: int = 640, target_fps: int = 30):
        self.handedness = handedness
        self.resize_long_side = resize_long_side
        self.target_fps = target_fps
        self.pose_estimator = PoseEstimator()
        self.metrics_calculator = MetricsCalculator(handedness=handedness)
        self.skill_grader = SkillGrader()
        self.report_generator = ReportGenerator()

    def process_video(self, video_path: str, handedness: str, skill_level: str, 
                      enable_bat_tracking: bool, enable_phase_detection: bool, 
                      progress_bar=None) -> Dict[str, Any]:
        """Process video, run analysis, and return results with annotated video path."""
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file."}

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_video_path = output_dir / f"annotated_{Path(video_path).name}"
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

        landmarks_history = []
        metrics_history = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            pose_results = self.pose_estimator.estimate_pose(frame)
            
            if pose_results and 'landmarks' in pose_results and pose_results.get('raw_results') and getattr(pose_results['raw_results'], 'pose_landmarks', None):
                landmarks_history.append(pose_results['landmarks'])
                # Simplified overlay for performance
                annotated_frame = frame.copy()
                self.pose_estimator.mp_drawing.draw_landmarks(
                    annotated_frame,
                    pose_results['raw_results'].pose_landmarks,
                    self.pose_estimator.mp_pose.POSE_CONNECTIONS
                )
                
                # Calculate metrics
                timestamp = frame_count / fps
                frame_metrics = self.metrics_calculator.calculate_frame_metrics(
                    pose_results['landmarks'], 
                    frame.shape[:2],  # Pass (height, width)
                    timestamp
                )
                metrics_history.append(frame_metrics)
                
            else:
                annotated_frame = frame
            
            out.write(annotated_frame)
            
            if progress_bar:
                progress = (frame_count + 1) / total_frames
                progress_text = f"Processing frame {frame_count + 1} of {total_frames}... ({int(progress * 100)}%)"
                progress_bar.progress(progress, text=progress_text)
            
            frame_count += 1

        cap.release()
        out.release()
        
        if not landmarks_history:
            return {"error": "No pose detected, cannot perform analysis."}

        # Perform full analysis if metrics were collected
        from utils.smoothing import Smoother

        smoother = Smoother()
        smoothness_score = smoother.compute_smoothness(metrics_history)

        if metrics_history:
            category_scores = self._compute_category_scores(metrics_history, skill_level)
            assessment = self.skill_grader.assess_skill_level(category_scores, {"smoothness": smoothness_score})
            evaluation = self.skill_grader.generate_skill_report(assessment)
            evaluation['global'] = {'avg_fps': fps}
        else:
            # Fallback to simplified analysis if no metrics
            evaluation = {
                'overall_score': 0.0, 
                'skill_level': 'N/A',
                'feedback': {'error': 'Could not compute metrics for evaluation.'},
                'global': {'avg_fps': fps}
            }
        
        report_path_html = output_dir / "report.html"
        self.report_generator.generate_html_report(
            {"evaluation": evaluation},  # ✅ wrap it
            str(report_path_html)
        )

        return {
            "annotated_video_path": str(output_video_path),
            "evaluation": evaluation,
            "report_paths": {"html": str(report_path_html)},
            "landmarks_data": landmarks_history
        }


# --- END INLINED ANALYSIS CLASS ---


# ---------------- MAIN APP ---------------- #

def main():
    load_dotenv()

    st.set_page_config(
        page_title="Cricket Cover Drive Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("🏏 Cricket Cover Drive Analysis")

    # Initialize Gemini Analyzer (optional)
    try:
        gemini_analyzer = GeminiAnalyzer(api_key=getenv("GEMINI_API_KEY", ""))
    except Exception as e:
        st.warning(f"Gemini analyzer not available: {e}")
        gemini_analyzer = None

    # Sidebar UI
    with st.sidebar:
        st.header("Analysis Settings")
        input_type = st.radio("Input Source", ["Upload Video File", "YouTube URL"])
        handedness = st.selectbox("Batter Handedness", ["right", "left"])
        resize_long_side = st.slider("Resize Long Side", 320, 1280, 640, 160)
        target_fps = st.slider("Target FPS", 15, 60, 30, 5)
        skill_level = st.selectbox("Reference Skill Level", ["beginner", "intermediate", "advanced", "professional"], index=1)
        enable_bat_tracking = st.checkbox("Enable Bat Tracking", value=True)
        enable_phase_detection = st.checkbox("Enable Phase Detection", value=True)

    # Main UI
    video_source = None
    if input_type == "Upload Video File":
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        if uploaded_file:
            video_path = Path("output") / uploaded_file.name
            Path("output").mkdir(exist_ok=True)
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            video_source = str(video_path)
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            st.video(video_source)
    else:  # YouTube URL
        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_url:
            with st.spinner("Downloading YouTube video..."):
                video_source = download_youtube_video(youtube_url, Path("output"))
            
            if video_source:
                st.success(f"✅ YouTube video downloaded: {Path(video_source).name}")
                st.video(video_source)
            else:
                st.error("Download failed. Please check the URL and try again.")

    if video_source and st.button("Start Analysis", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Analysis starting...")
        results = run_analysis(
            video_source, handedness, resize_long_side, target_fps,
            skill_level, enable_bat_tracking, enable_phase_detection,
            progress_bar
        )
        if results:
            st.session_state['analysis_results'] = results
            st.success("✅ Analysis complete!")
        else:
            st.error("❌ Analysis failed.")

    if 'analysis_results' in st.session_state:
            display_results(st.session_state['analysis_results'], gemini_analyzer)


# ---------------- RUN ANALYSIS ---------------- #

def run_analysis(video_source, handedness, resize_long_side, target_fps, skill_level, enable_bat_tracking, enable_phase_detection, progress_bar):
    try:
        analyzer = CoverDriveAnalyzer(
            handedness=handedness,
            resize_long_side=resize_long_side,
            target_fps=target_fps
        )

        Path("output").mkdir(exist_ok=True)
        results = analyzer.process_video(
            video_path=video_source,
            handedness=handedness,
            skill_level=skill_level,
            enable_bat_tracking=enable_bat_tracking,
            enable_phase_detection=enable_phase_detection,
            progress_bar=progress_bar
        )

        # Ensure annotated video path is present (fallback if analyzer didn't attach it)
        if not results.get("annotated_video_path"):
            default_path = Path("output") / "annotated_video.mp4"
            if default_path.exists():
                results["annotated_video_path"] = str(default_path)

        # Generate HTML report if not already set
        if not results.get("report_paths", {}).get("html"):
            try:
                report_gen = ReportGenerator()
                html_report_path = Path("output") / "report.html"
                report_gen.generate_html_report(results, str(html_report_path))
                results.setdefault("report_paths", {})["html"] = str(html_report_path)
            except Exception as e:
                st.warning(f"Report generation failed: {e}")

        return results

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None


# ---------------- DISPLAY RESULTS ---------------- #

def display_results(results, gemini_analyzer):
    st.header("Analysis Results")

    # Download button for the report
    report_path = results.get("report_paths", {}).get("html")
    if report_path and Path(report_path).exists():
        with open(report_path, "rb") as f:
            st.download_button(
                label="📄 Download Full Report",
                data=f,
                file_name="cover_drive_analysis_report.html",
                mime="text/html",
                use_container_width=True
            )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📹 Annotated Video")
        display_video_results(results)

    with col2:
        st.subheader("📊 Performance Overview")
        display_overview(results)
        
        st.subheader("📈 Category Breakdown")
        display_category_scores(results)



def display_overview(results: Dict[str, Any]):
    eval_data = results.get('evaluation', {})
    if not eval_data:
        st.warning("No evaluation data available.")
        return

    col1, col2 = st.columns(2)
    col1.metric("Overall Score", f"{eval_data.get('overall_score', 0):.1f}/10")
    col2.metric("Skill Level", str(eval_data.get('skill_level', 'N/A')).title())


def display_category_scores(results: Dict[str, Any]):
    """Displays the breakdown of scores by category."""
    eval_data = results.get('evaluation', {})
    category_scores = eval_data.get('category_breakdown')

    if not category_scores:
        st.info("No category breakdown available.")
        return

    for category, score in category_scores.items():
        st.slider(category, 0.0, 10.0, float(score), disabled=True)




def display_reports(results: Dict[str, Any]):
    path = results.get("report_paths", {}).get("html")
    if not path or not Path(path).exists():
        st.info("No HTML report available.")
        return
    try:
        with open(path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=800, scrolling=True)
    except Exception as e:
        st.error(f"Error loading report: {e}")


def display_video_results(results: Dict[str, Any]):
    st.subheader("📹 Videos")
    annotated_video_path = results.get("annotated_video_path")

    if annotated_video_path and Path(annotated_video_path).exists():
        try:
            with open(annotated_video_path, "rb") as f:
                video_bytes = f.read()
            st.video(video_bytes, format="video/mp4")
        except Exception as e:
            st.error(f"Error displaying annotated video: {e}")
    else:
        st.warning("Annotated video not available.")




def download_youtube_video(url: str, output_dir: Path) -> Optional[str]:
    """Downloads a video from YouTube and returns the local file path."""
    try:
        output_dir.mkdir(exist_ok=True)
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_dir / '%(id)s.%(ext)s'),
            'merge_output_format': 'mp4',
            'quiet': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None


if __name__ == "__main__":
    main()
