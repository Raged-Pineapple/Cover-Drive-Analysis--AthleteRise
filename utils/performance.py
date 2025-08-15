"""
Performance monitoring and optimization for real-time cricket analysis.
Tracks FPS, optimizes processing, and provides performance recommendations.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import deque
import threading
import queue

import cv2


@dataclass
class PerformanceMetrics:
    """Performance metrics for a processing session."""
    avg_fps: float
    min_fps: float
    max_fps: float
    frame_processing_times: List[float]
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    optimization_applied: List[str] = None


class PerformanceMonitor:
    """Monitors and tracks real-time performance metrics."""
    
    def __init__(self, target_fps: float = 30.0, window_size: int = 30):
        self.target_fps = target_fps
        self.window_size = window_size
        
        # Performance tracking
        self.frame_times = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.start_time = None
        self.frame_count = 0
        
        # Performance thresholds
        self.thresholds = {
            'min_acceptable_fps': 10.0,
            'optimal_fps': 25.0,
            'max_processing_time': 0.1,  # 100ms per frame
            'frame_skip_threshold': 0.15  # Skip frames if processing takes too long
        }
        
        # Optimization state
        self.optimizations_applied = []
        self.frame_skip_count = 0
        self.resolution_reduced = False
        
    def start_session(self):
        """Start a new processing session."""
        self.start_time = time.time()
        self.frame_count = 0
        self.frame_times.clear()
        self.processing_times.clear()
        self.optimizations_applied.clear()
        self.frame_skip_count = 0
        self.resolution_reduced = False
    
    def record_frame(self, processing_time: float):
        """Record processing time for a frame."""
        current_time = time.time()
        
        if self.start_time is None:
            self.start_session()
        
        # Record processing time
        self.processing_times.append(processing_time)
        
        # Calculate frame time (time between frames)
        if len(self.frame_times) > 0:
            frame_time = current_time - self.frame_times[-1]
        else:
            frame_time = processing_time
        
        self.frame_times.append(current_time)
        self.frame_count += 1
    
    def get_current_fps(self) -> float:
        """Get current FPS based on recent frames."""
        if len(self.frame_times) < 2:
            return 0.0
        
        # Calculate FPS from recent frame times
        recent_times = list(self.frame_times)[-min(10, len(self.frame_times)):]
        if len(recent_times) < 2:
            return 0.0
        
        frame_intervals = [recent_times[i] - recent_times[i-1] for i in range(1, len(recent_times))]
        avg_interval = np.mean(frame_intervals)
        
        return 1.0 / avg_interval if avg_interval > 0 else 0.0
    
    def get_performance_summary(self) -> PerformanceMetrics:
        """Get comprehensive performance summary."""
        if not self.processing_times:
            return PerformanceMetrics(
                avg_fps=0.0, min_fps=0.0, max_fps=0.0,
                frame_processing_times=[], optimization_applied=[]
            )
        
        # Calculate FPS statistics
        fps_values = []
        for i in range(1, len(self.frame_times)):
            interval = self.frame_times[i] - self.frame_times[i-1]
            if interval > 0:
                fps_values.append(1.0 / interval)
        
        if not fps_values:
            avg_fps = min_fps = max_fps = 0.0
        else:
            avg_fps = np.mean(fps_values)
            min_fps = np.min(fps_values)
            max_fps = np.max(fps_values)
        
        return PerformanceMetrics(
            avg_fps=avg_fps,
            min_fps=min_fps,
            max_fps=max_fps,
            frame_processing_times=list(self.processing_times),
            optimization_applied=self.optimizations_applied.copy()
        )
    
    def should_skip_frame(self) -> bool:
        """Determine if current frame should be skipped for performance."""
        if len(self.processing_times) == 0:
            return False
        
        avg_processing_time = np.mean(self.processing_times)
        current_fps = self.get_current_fps()
        
        # Skip frame if processing is too slow or FPS is too low
        should_skip = (
            avg_processing_time > self.thresholds['max_processing_time'] or
            current_fps < self.thresholds['min_acceptable_fps']
        )
        
        if should_skip:
            self.frame_skip_count += 1
        
        return should_skip
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for performance optimization."""
        recommendations = []
        current_fps = self.get_current_fps()
        
        if current_fps < self.thresholds['min_acceptable_fps']:
            recommendations.append("FPS too low - consider reducing resolution")
            recommendations.append("FPS too low - consider using lighter pose model")
        
        if len(self.processing_times) > 0:
            avg_processing_time = np.mean(self.processing_times)
            if avg_processing_time > self.thresholds['max_processing_time']:
                recommendations.append("Processing time too high - optimize algorithms")
        
        if self.frame_skip_count > 0:
            recommendations.append(f"Skipped {self.frame_skip_count} frames - reduce processing load")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable range")
        
        return recommendations


class PerformanceOptimizer:
    """Applies performance optimizations based on monitoring data."""
    
    def __init__(self):
        self.optimization_levels = {
            'none': {'resize_factor': 1.0, 'model_complexity': 2, 'smoothing_window': 7},
            'light': {'resize_factor': 0.8, 'model_complexity': 1, 'smoothing_window': 5},
            'medium': {'resize_factor': 0.6, 'model_complexity': 0, 'smoothing_window': 3},
            'heavy': {'resize_factor': 0.4, 'model_complexity': 0, 'smoothing_window': 1}
        }
        
        self.current_level = 'none'
        self.optimization_history = []
    
    def get_optimization_level(self, current_fps: float, target_fps: float) -> str:
        """Determine appropriate optimization level based on performance."""
        fps_ratio = current_fps / target_fps if target_fps > 0 else 1.0
        
        if fps_ratio >= 0.9:
            return 'none'
        elif fps_ratio >= 0.7:
            return 'light'
        elif fps_ratio >= 0.5:
            return 'medium'
        else:
            return 'heavy'
    
    def apply_optimizations(self, monitor: PerformanceMonitor, 
                          current_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimizations based on current performance."""
        current_fps = monitor.get_current_fps()
        target_fps = monitor.target_fps
        
        # Determine optimization level
        new_level = self.get_optimization_level(current_fps, target_fps)
        
        if new_level != self.current_level:
            self.optimization_history.append({
                'from_level': self.current_level,
                'to_level': new_level,
                'fps': current_fps,
                'timestamp': time.time()
            })
            self.current_level = new_level
        
        # Get optimization settings
        optimizations = self.optimization_levels[new_level]
        
        # Apply optimizations
        optimized_settings = current_settings.copy()
        
        # Resize optimization
        if 'resize_long_side' in optimized_settings:
            original_size = optimized_settings['resize_long_side']
            optimized_settings['resize_long_side'] = int(original_size * optimizations['resize_factor'])
        
        # Model complexity optimization
        if 'model_complexity' in optimized_settings:
            optimized_settings['model_complexity'] = optimizations['model_complexity']
        
        # Smoothing optimization
        if 'smoothing_window' in optimized_settings:
            optimized_settings['smoothing_window'] = optimizations['smoothing_window']
        
        # Add optimization info
        optimized_settings['optimization_level'] = new_level
        optimized_settings['optimization_applied'] = list(optimizations.keys())
        
        return optimized_settings
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of applied optimizations."""
        return {
            'current_level': self.current_level,
            'optimization_history': self.optimization_history,
            'total_optimizations': len(self.optimization_history)
        }


class AsyncVideoWriter:
    """Asynchronous video writer for improved performance."""
    
    def __init__(self, output_path: str, fps: float, frame_size: Tuple[int, int]):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        
        # Async processing
        self.frame_queue = queue.Queue(maxsize=100)
        self.running = False
        self.worker_thread = None
        
        # Statistics
        self.frames_written = 0
        self.frames_dropped = 0
    
    def start(self):
        """Start the async video writer."""
        self.running = True
        self.worker_thread = threading.Thread(target=self._write_frames)
        self.worker_thread.start()
    
    def stop(self):
        """Stop the async video writer."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        self.writer.release()
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """Queue a frame for writing."""
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False
    
    def _write_frames(self):
        """Worker thread for writing frames."""
        while self.running or not self.frame_queue.empty():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.writer.write(frame)
                self.frames_written += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error writing frame: {e}")
                break


class PerformanceLogger:
    """Logs performance metrics to console and file."""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file
        self.log_interval = 30  # Log every 30 frames
        self.frame_count = 0
    
    def log_performance(self, monitor: PerformanceMonitor, 
                       optimizer: PerformanceOptimizer):
        """Log current performance metrics."""
        self.frame_count += 1
        
        if self.frame_count % self.log_interval == 0:
            current_fps = monitor.get_current_fps()
            performance_summary = monitor.get_performance_summary()
            optimization_summary = optimizer.get_optimization_summary()
            
            # Console output
            print(f"\n=== Performance Log (Frame {self.frame_count}) ===")
            print(f"Current FPS: {current_fps:.1f}")
            print(f"Average FPS: {performance_summary.avg_fps:.1f}")
            print(f"Min/Max FPS: {performance_summary.min_fps:.1f}/{performance_summary.max_fps:.1f}")
            print(f"Optimization Level: {optimization_summary['current_level']}")
            print(f"Frames Skipped: {monitor.frame_skip_count}")
            
            if performance_summary.frame_processing_times:
                avg_processing = np.mean(performance_summary.frame_processing_times)
                print(f"Average Processing Time: {avg_processing*1000:.1f}ms")
            
            # Recommendations
            recommendations = monitor.get_optimization_recommendations()
            if recommendations:
                print("Recommendations:")
                for rec in recommendations:
                    print(f"  - {rec}")
            
            # File logging
            if self.log_file:
                self._write_to_file(performance_summary, optimization_summary)
    
    def _write_to_file(self, performance_summary: PerformanceMetrics, 
                      optimization_summary: Dict[str, Any]):
        """Write performance data to log file."""
        try:
            with open(self.log_file, 'a') as f:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{timestamp}, {performance_summary.avg_fps:.2f}, "
                       f"{performance_summary.min_fps:.2f}, {performance_summary.max_fps:.2f}, "
                       f"{optimization_summary['current_level']}, {len(performance_summary.frame_processing_times)}\n")
        except Exception as e:
            print(f"Error writing to log file: {e}")


def create_performance_monitor(target_fps: float = 30.0) -> PerformanceMonitor:
    """Create a performance monitor with default settings."""
    return PerformanceMonitor(target_fps=target_fps)


def create_performance_optimizer() -> PerformanceOptimizer:
    """Create a performance optimizer with default settings."""
    return PerformanceOptimizer()


def create_async_video_writer(output_path: str, fps: float, 
                            frame_size: Tuple[int, int]) -> AsyncVideoWriter:
    """Create an async video writer."""
    return AsyncVideoWriter(output_path, fps, frame_size)
