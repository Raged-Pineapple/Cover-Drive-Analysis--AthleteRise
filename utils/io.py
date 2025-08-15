"""
Video I/O utilities for reading from files/URLs and writing annotated videos.
Handles YouTube downloads and video processing operations.
"""

import cv2
import numpy as np
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union, List
import requests
from urllib.parse import urlparse
import re


class VideoIO:
    """Handles video input/output operations including YouTube downloads."""
    
    def __init__(self, temp_dir: Optional[str] = None):
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir())
        self.temp_dir.mkdir(exist_ok=True)
        
        # Supported video formats
        self.supported_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
    def get_video_capture(self, source: str, resize_long_side: Optional[int] = None) -> Optional[cv2.VideoCapture]:
        """Get OpenCV VideoCapture from file or URL."""
        
        processed_source = self._process_source(source)
        if not processed_source:
            return None
        
        # Try to open video
        cap = cv2.VideoCapture(processed_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source: {source}")
            return None
        
        # Apply resizing if requested
        if resize_long_side:
            self._configure_capture_size(cap, resize_long_side)
        
        return cap
    
    def _process_source(self, source: str) -> Optional[str]:
        """Process source (file path or URL) and return usable path."""
        
        # Check if it's a YouTube URL
        if self._is_youtube_url(source):
            return self._download_youtube_video(source)
        
        # Check if it's a local file
        elif os.path.isfile(source):
            return source
        
        # Check if it's a direct video URL
        elif self._is_video_url(source):
            return self._download_video_from_url(source)
        
        # Check if it's a webcam index
        elif source.isdigit():
            return int(source)
        
        print(f"Error: Unrecognized video source format: {source}")
        return None
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is a YouTube video."""
        youtube_patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/shorts/([^&\n?#]+)'
        ]
        
        for pattern in youtube_patterns:
            if re.search(pattern, url):
                return True
        return False
    
    def _is_video_url(self, url: str) -> bool:
        """Check if URL points to a video file."""
        try:
            parsed = urlparse(url)
            path = parsed.path.lower()
            return any(path.endswith(ext) for ext in self.supported_formats)
        except:
            return False
    
    def _download_youtube_video(self, url: str) -> Optional[str]:
        """Download YouTube video using yt-dlp."""
        
        try:
            # Check if yt-dlp is available
            subprocess.run(['yt-dlp', '--version'], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: yt-dlp not found. Please install it: pip install yt-dlp")
            return None
        
        # Create temporary file path
        temp_file = self.temp_dir / f"youtube_video_{hash(url) % 1000000}.mp4"
        
        # Download video
        try:
            cmd = [
                'yt-dlp',
                '-f', 'best[height<=720]/best',  # Limit quality for faster processing
                '-o', str(temp_file),
                '--no-playlist',
                url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Error downloading YouTube video: {result.stderr}")
                return None
                
            if temp_file.exists():
                return str(temp_file)
            else:
                print("Error: Downloaded file not found")
                return None
                
        except subprocess.TimeoutExpired:
            print("Error: YouTube download timed out")
            return None
        except Exception as e:
            print(f"Error downloading YouTube video: {e}")
            return None
    
    def _download_video_from_url(self, url: str) -> Optional[str]:
        """Download video from direct URL."""
        
        try:
            # Create temporary file
            temp_file = self.temp_dir / f"downloaded_video_{hash(url) % 1000000}.mp4"
            
            # Download with streaming
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(temp_file)
            
        except Exception as e:
            print(f"Error downloading video from URL: {e}")
            return None
    
    def _configure_capture_size(self, cap: cv2.VideoCapture, resize_long_side: int):
        """Configure video capture with resizing."""
        
        # Get original dimensions
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate new dimensions
        if original_width > original_height:
            new_width = resize_long_side
            new_height = int(original_height * resize_long_side / original_width)
        else:
            new_height = resize_long_side
            new_width = int(original_width * resize_long_side / original_height)
        
        # Note: We'll resize frames during processing since VideoCapture
        # resizing isn't reliable across different backends
        self._target_size = (new_width, new_height)
    
    def create_video_writer(self, output_path: str, fps: float, 
                          frame_size: Tuple[int, int], 
                          codec: str = 'mp4v') -> cv2.VideoWriter:
        """Create OpenCV VideoWriter for output."""
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create video writer
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        if not out.isOpened():
            raise RuntimeError(f"Could not create video writer for: {output_path}")
        
        return out
    
    def resize_frame(self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize frame to target dimensions."""
        
        if target_size is None and hasattr(self, '_target_size'):
            target_size = self._target_size
        
        if target_size is None:
            return frame
        
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    def get_video_info(self, source: str) -> Optional[dict]:
        """Get video information (duration, fps, resolution, etc.)."""
        
        cap = self.get_video_capture(source)
        if cap is None:
            return None
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'codec': int(cap.get(cv2.CAP_PROP_FOURCC))
        }
        
        cap.release()
        return info
    
    def extract_frames(self, source: str, output_dir: str, 
                      frame_interval: int = 30, max_frames: Optional[int] = None) -> List[str]:
        """Extract frames from video for debugging/analysis."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        cap = self.get_video_capture(source)
        if cap is None:
            return []
        
        frame_paths = []
        frame_idx = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at specified intervals
            if frame_idx % frame_interval == 0:
                frame_filename = output_path / f"frame_{frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                frame_paths.append(str(frame_filename))
                saved_count += 1
                
                if max_frames and saved_count >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        return frame_paths
    
    def create_video_from_frames(self, frame_paths: List[str], 
                                output_path: str, fps: float = 30.0) -> bool:
        """Create video from sequence of frame images."""
        
        if not frame_paths:
            return False
        
        # Get frame dimensions from first frame
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            return False
        
        height, width = first_frame.shape[:2]
        
        # Create video writer
        try:
            writer = self.create_video_writer(output_path, fps, (width, height))
            
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    writer.write(frame)
            
            writer.release()
            return True
            
        except Exception as e:
            print(f"Error creating video from frames: {e}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary downloaded files."""
        
        try:
            for file_path in self.temp_dir.glob("youtube_video_*.mp4"):
                file_path.unlink()
            for file_path in self.temp_dir.glob("downloaded_video_*.mp4"):
                file_path.unlink()
        except Exception as e:
            print(f"Warning: Could not clean up temp files: {e}")


class BatchVideoProcessor:
    """Process multiple videos in batch mode."""
    
    def __init__(self):
        self.video_io = VideoIO()
        
    def process_video_list(self, video_sources: List[str], 
                          output_dir: str, 
                          analyzer_config: dict) -> List[dict]:
        """Process multiple videos and return results."""
        
        results = []
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, source in enumerate(video_sources):
            print(f"Processing video {i+1}/{len(video_sources)}: {source}")
            
            try:
                # Create individual output directory
                video_output_dir = output_path / f"video_{i+1:03d}"
                video_output_dir.mkdir(exist_ok=True)
                
                # Process video (would call main analyzer here)
                # result = analyzer.process_video(source, str(video_output_dir))
                # results.append(result)
                
            except Exception as e:
                print(f"Error processing {source}: {e}")
                results.append({"error": str(e), "source": source})
        
        return results
    
    def create_batch_report(self, results: List[dict], output_path: str):
        """Create summary report for batch processing."""
        
        # Aggregate statistics
        total_videos = len(results)
        successful = sum(1 for r in results if "error" not in r)
        failed = total_videos - successful
        
        # Calculate average scores
        if successful > 0:
            avg_scores = {}
            for category in ['Footwork', 'Head Position', 'Swing Control', 'Balance', 'Follow-through']:
                scores = []
                for result in results:
                    if "error" not in result and "summary" in result:
                        scores.append(result["summary"][category]["score"])
                
                if scores:
                    avg_scores[category] = sum(scores) / len(scores)
        
        # Create report
        report = {
            "batch_summary": {
                "total_videos": total_videos,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_videos if total_videos > 0 else 0
            },
            "average_scores": avg_scores if successful > 0 else {},
            "individual_results": results
        }
        
        # Save report
        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)


class VideoValidator:
    """Validate video files and sources before processing."""
    
    def __init__(self):
        self.video_io = VideoIO()
    
    def validate_source(self, source: str) -> dict:
        """Validate video source and return status."""
        
        validation = {
            "valid": False,
            "source": source,
            "type": "unknown",
            "issues": [],
            "info": {}
        }
        
        try:
            # Determine source type
            if self.video_io._is_youtube_url(source):
                validation["type"] = "youtube"
            elif os.path.isfile(source):
                validation["type"] = "local_file"
            elif self.video_io._is_video_url(source):
                validation["type"] = "url"
            elif source.isdigit():
                validation["type"] = "webcam"
            else:
                validation["issues"].append("Unknown source type")
                return validation
            
            # Try to get video info
            info = self.video_io.get_video_info(source)
            if info is None:
                validation["issues"].append("Could not open video source")
                return validation
            
            validation["info"] = info
            
            # Check video properties
            if info["width"] < 320 or info["height"] < 240:
                validation["issues"].append("Video resolution too low")
            
            if info["fps"] < 10:
                validation["issues"].append("Frame rate too low")
            
            if info["frame_count"] < 30:
                validation["issues"].append("Video too short")
            
            # If no major issues, mark as valid
            validation["valid"] = len(validation["issues"]) == 0
            
        except Exception as e:
            validation["issues"].append(f"Validation error: {str(e)}")
        
        return validation
    
    def batch_validate(self, sources: List[str]) -> List[dict]:
        """Validate multiple video sources."""
        
        results = []
        for source in sources:
            result = self.validate_source(source)
            results.append(result)
        
        return results