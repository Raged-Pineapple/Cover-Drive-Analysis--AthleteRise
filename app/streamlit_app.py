"""
Streamlit web application for Cricket Cover Drive Analysis.
Provides interactive interface for video upload, processing, and results display.
"""
import warnings
# Suppress noisy protobuf deprecation warning from dependencies
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype() is deprecated",
    category=UserWarning,
    module="google.protobuf.symbol_database"
)

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
import subprocess
load_dotenv()
# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent.resolve())
sys.path.insert(0, parent_dir)

# Import from utils
from utils.gemini_analysis import GeminiAnalyzer
# from utils.ai_helper import extract_text
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
        # Write to a robust intermediate AVI (MJPG), then convert to H.264 MP4 for browsers
        intermediate_path = output_dir / f"annotated_{Path(video_path).stem}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(str(intermediate_path), fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            return {"error": "Failed to open video writer. MJPG codec not available. Install opencv-contrib or fallback codecs."}

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

        # Quick sanity check: ensure file was written and non-empty
        try:
            if intermediate_path.exists() and intermediate_path.stat().st_size < 1024:
                return {"error": "Annotated video file appears empty or too small. Encoding likely failed."}
        except Exception:
            pass
        
        # Optional: create a browser-friendly copy using H.264 if moviepy/ffmpeg is available
        web_safe_path = None
        encode_notes = []
        try:
            from moviepy.editor import VideoFileClip
            web_safe_path = output_dir / f"annotated_{Path(video_path).stem}.mp4"
            # Re-encode to H.264 for maximum browser compatibility
            # Disable audio, set fps explicitly
            clip = VideoFileClip(str(intermediate_path))
            clip.write_videofile(
                str(web_safe_path),
                codec="libx264",
                audio=False,
                fps=fps,
                preset="ultrafast",
                ffmpeg_params=[
                    "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-crf", "22",
                ],
                verbose=True,
            )
            clip.close()
            # Validate MP4 has readable frames
            try:
                vcap = cv2.VideoCapture(str(web_safe_path))
                test_ret, _ = vcap.read()
                frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
                vcap.release()
                if not test_ret or frames == 0:
                    encode_notes.append("Validation failed: MP4 has no readable frames; falling back to AVI.")
                    web_safe_path = None
            except Exception as ve:
                encode_notes.append(f"Validation error: {ve}")
        except Exception as _e:
            # Non-fatal; we will use the original annotated video instead
            encode_notes.append(f"Conversion failed: {_e}")
            web_safe_path = None

        # Fallback: try direct ffmpeg CLI via imageio-ffmpeg, if MoviePy path failed
        if web_safe_path is None:
            try:
                import imageio_ffmpeg
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                web_safe_path = output_dir / f"annotated_{Path(video_path).stem}.mp4"
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i", str(intermediate_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-crf", "22",
                    "-r", str(fps if fps > 0 else 30),
                    str(web_safe_path)
                ]
                completed = subprocess.run(cmd, capture_output=True, text=True)
                if completed.returncode != 0:
                    encode_notes.append(f"ffmpeg fallback failed: {completed.stderr[:300]}")
                    web_safe_path = None
                else:
                    # Validate produced MP4
                    vcap = cv2.VideoCapture(str(web_safe_path))
                    ok, _ = vcap.read()
                    frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
                    vcap.release()
                    if not ok or frames == 0:
                        encode_notes.append("ffmpeg output not readable; falling back to AVI.")
                        web_safe_path = None
                    else:
                        encode_notes.append("ffmpeg fallback succeeded.")
            except Exception as fe:
                encode_notes.append(f"ffmpeg fallback error: {fe}")
        
        # Prefer the web-safe mp4; if unavailable, fall back to the intermediate AVI
        output_video_path = web_safe_path if web_safe_path else intermediate_path
        
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

        # ✅ NEW: AI Recommendations via Gemini
        try:
            from utils.ai_helper import generate_ai_recommendations  # create this helper

            evaluation['recommendations'] = generate_ai_recommendations(evaluation)
        except Exception as e:
            print(f"[AI Recommendations] Failed: {e}")
            evaluation['recommendations'] = ["No AI recommendations available."]

        results = {
            "annotated_video_path": str(output_video_path),
            "web_annotated_video_path": str(web_safe_path) if web_safe_path else None,
            "intermediate_video_path": str(intermediate_path),
            "encode_notes": encode_notes,
            "evaluation": evaluation,
            "metrics_history": metrics_history,
            "frames_analyzed": len(metrics_history),
            "smoothness_score": smoothness_score,
            "landmarks_data": landmarks_history,
        }

        report_path_html = output_dir / "report.html"
        self.report_generator.generate_html_report(results, str(report_path_html))
        results["report_paths"] = {"html": str(report_path_html)}
        # Try creating a PDF as well (if dependencies available)
        try:
            report_path_pdf = output_dir / "report.pdf"
            self.report_generator.generate_pdf_report(results, str(report_path_pdf))
            results["report_paths"]["pdf"] = str(report_path_pdf)
        except Exception as e:
            # Non-fatal: PDF backend not available
            print(f"[ReportGen] PDF not generated: {e}")

        return results



# --- END INLINED ANALYSIS CLASS ---


# ---------------- MAIN APP ---------------- #

def main():
    

    st.set_page_config(
        page_title="Cricket Cover Drive Analysis",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.title("Cricket Cover Drive Analysis")

    # Initialize session state for persistence across reruns
    st.session_state.setdefault('analysis_results', None)
    st.session_state.setdefault('last_video_source', None)
    st.session_state.setdefault('debug_mode', False)

    # # Initialize Gemini Analyzer (optional)
    # try:
    #     gemini_analyzer = GeminiAnalyzer(api_key=getenv("GEMINI_API_KEY", ""))
    # except Exception as e:
    #     st.warning(f"Gemini analyzer not available: {e}")
    #     gemini_analyzer = None

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
        debug_mode = st.checkbox("Debug mode", value=bool(st.session_state.get('debug_mode', False)))
        # Persist debug mode across reruns
        st.session_state['debug_mode'] = bool(debug_mode)

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
            # Reset results if video changes
            if st.session_state.get('last_video_source') != video_source:
                st.session_state['analysis_results'] = None
                st.session_state['last_video_source'] = video_source
            st.success(f"Video uploaded: {uploaded_file.name}")
            st.video(video_source)
    else:  # YouTube URL
        youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        if youtube_url:
            with st.spinner("Downloading YouTube video..."):
                video_source = download_youtube_video(youtube_url, Path("output"))
            
            if video_source:
                # Reset results if video changes
                if st.session_state.get('last_video_source') != video_source:
                    st.session_state['analysis_results'] = None
                    st.session_state['last_video_source'] = video_source
                st.success(f"YouTube video downloaded: {Path(video_source).name}")
                st.video(video_source)
            else:
                st.error("Download failed. Please check the URL and try again.")

    # If we have previous results, show them automatically (so toggles/downloads don't lose state)
    if st.session_state.get('analysis_results') and not st.session_state.get('just_ran'):
        display_results(st.session_state['analysis_results'])

    if video_source and st.button("Start Analysis", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Analysis starting...")
        results = run_analysis(
            video_source, handedness, resize_long_side, target_fps,
            skill_level, enable_bat_tracking, enable_phase_detection,
            progress_bar
        )
        if results and not results.get("error"):
            st.session_state['analysis_results'] = results
            st.session_state['just_ran'] = True
            st.success("Analysis complete!")
            # Immediately render results
            display_results(results)
            # Clear the just_ran flag after showing once
            st.session_state['just_ran'] = False
        else:
            err = results.get("error") if isinstance(results, dict) else None
            st.error(f"Analysis failed. {err or ''}")

    


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

def display_results(results):
    st.header("Analysis Results")

    # Download buttons for reports
    report_paths = results.get("report_paths", {})
    report_html = report_paths.get("html")
    report_pdf = report_paths.get("pdf")
    if report_html and Path(report_html).exists():
        with open(report_html, "rb") as f:
            st.download_button(
                label="Download Full Report (HTML)",
                data=f,
                file_name="cover_drive_analysis_report.html",
                mime="text/html",
                use_container_width=True
            )
    if report_pdf and Path(report_pdf).exists():
        with open(report_pdf, "rb") as fpdf:
            st.download_button(
                label="Download as PDF",
                data=fpdf,
                file_name="cover_drive_analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    elif report_html and Path(report_html).exists() and not report_pdf:
        st.caption("PDF generation not available (install pdfkit with wkhtmltopdf or reportlab).")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Annotated Video")
        display_video_results(results)

    with col2:
        st.subheader("Performance Overview")
        display_overview(results)
        
        st.subheader("Category Breakdown")
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
        # Use a small float step to avoid min/max/step conflicts with float defaults
        st.slider(category, 0.0, 10.0, float(score), step=0.01, disabled=True)




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
    st.subheader("Videos")
    web_safe = results.get("web_annotated_video_path")
    annotated_video_path = web_safe or results.get("annotated_video_path")

    if annotated_video_path and Path(annotated_video_path).exists():
        try:
            # If we only have an AVI fallback, browsers will not play it inline
            if Path(annotated_video_path).suffix.lower() == ".avi":
                st.info("Embedded playback not supported for AVI. Please download and play locally.")
                with open(annotated_video_path, "rb") as f:
                    st.download_button("Download annotated video (AVI)", f, file_name=Path(annotated_video_path).name, mime="video/avi")
                # Offer one-click conversion to MP4 without re-running analysis
                fps_hint = results.get("evaluation", {}).get("global", {}).get("avg_fps", 30)
                if st.button("Convert to MP4 (H.264)", use_container_width=True):
                    try:
                        import imageio_ffmpeg, subprocess
                        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                        src = Path(results.get("intermediate_video_path") or annotated_video_path)
                        dst = Path(src).with_suffix('.mp4')
                        cmd = [
                            ffmpeg_exe,
                            "-y",
                            "-i", str(src),
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart",
                            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                            "-crf", "22",
                            "-r", str(int(fps_hint) if fps_hint else 30),
                            str(dst)
                        ]
                        completed = subprocess.run(cmd, capture_output=True, text=True)
                        if completed.returncode == 0 and dst.exists():
                            st.session_state['analysis_results']['web_annotated_video_path'] = str(dst)
                            st.success("MP4 created. Reloading...")
                            st.experimental_rerun()
                        else:
                            st.error("MP4 conversion failed. See debug notes.")
                    except Exception as ce:
                        st.error(f"Conversion error: {ce}")
            else:
                # Use file path for Streamlit to infer and serve correctly
                st.video(str(annotated_video_path))
                if web_safe:
                    st.caption("Playing web-optimized H.264 copy for maximum compatibility.")
        except Exception as e:
            st.error(f"Error displaying annotated video: {e}")
            st.info("You can still download and play the file locally:")
            with open(annotated_video_path, "rb") as f:
                st.download_button("Download annotated video", f, file_name=Path(annotated_video_path).name, mime="video/mp4")
    else:
        st.warning("Annotated video not available.")

    # Always show diagnostics in debug mode
    if st.session_state.get('debug_mode'):
        p = Path(web_safe or results.get("annotated_video_path") or "")
        st.markdown("**Debug: Video diagnostics**")
        st.write({
            "web_annotated_video_path": web_safe,
            "annotated_video_path": results.get("annotated_video_path"),
            "intermediate_video_path": results.get("intermediate_video_path"),
            "encode_notes": results.get("encode_notes"),
            "exists": p.exists() if p else False,
            "suffix": p.suffix if p else None,
            "size_bytes": (p.stat().st_size if p.exists() else None) if p else None,
        })
        if p and p.exists():
            try:
                cap = cv2.VideoCapture(str(p))
                fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                codec = ''.join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ret, frame = cap.read()
                cap.release()
                st.write({
                    "cv2_codec": codec,
                    "fps": fps,
                    "frames": frame_count,
                    "width": width,
                    "height": height,
                    "first_frame_readable": bool(ret),
                })
            except Exception as e:
                st.warning(f"Debug: could not probe video with OpenCV: {e}")
        if p and p.exists():
            try:
                with open(p, "rb") as f:
                    head = f.read(12)
                st.write({"file_header_hex": head.hex()})
            except Exception:
                pass

        # Always provide a download for manual check
        if p and p.exists():
            with open(p, "rb") as f:
                st.download_button("Download (debug)", f, file_name=p.name, mime="video/mp4")


def download_youtube_video(url: str, output_dir: Path) -> Optional[str]:
    """Downloads a video from YouTube and returns the local file path (.mp4)."""
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
            filename = ydl.prepare_filename(info)

        # If the merger produced an mp4 with different suffix handling, prefer that
        p = Path(filename)
        mp4_candidate = output_dir / f"{p.stem}.mp4"
        if p.suffix.lower() != '.mp4' and mp4_candidate.exists():
            return str(mp4_candidate)
        return str(p)
    except Exception as e:
        st.error(f"Failed to download video: {e}")
        return None


if __name__ == "__main__":
    main()
