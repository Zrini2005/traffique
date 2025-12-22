#!/usr/bin/env python3
"""
Advanced Vehicle Trajectory Tracker for Drone Videos
Handles noise, jitter, and creates smooth trajectory visualizations

Features:
- Multi-level noise reduction (Kalman + Savitzky-Golay + Moving Average)
- Outlier detection and removal
- Smooth trajectory drawing
- Individual and combined visualizations
"""

import csv
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from collections import defaultdict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import json

# Import existing modules
from interactive_analytics import VehicleAnalyzer, VehicleTracker


class TrajectorySmootherAdvanced:
    """Advanced trajectory smoothing with multiple noise reduction techniques"""
    
    @staticmethod
    def remove_outliers(trajectory: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Remove outlier points using velocity-based detection"""
        
        if len(trajectory) < 3:
            return trajectory
        
        # Calculate velocities
        velocities = np.diff(trajectory, axis=0)
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Detect outliers using MAD (Median Absolute Deviation)
        median_speed = np.median(speeds)
        mad = np.median(np.abs(speeds - median_speed))
        
        if mad < 1e-6:
            return trajectory
        
        # Mark outliers
        threshold_speed = median_speed + threshold * mad * 1.4826  # MAD to std conversion
        
        # Keep first point always
        cleaned = [trajectory[0]]
        
        for i in range(1, len(trajectory)):
            if i < len(speeds) and speeds[i-1] > threshold_speed:
                # Outlier detected, interpolate
                if len(cleaned) > 0:
                    # Use previous valid point
                    cleaned.append(cleaned[-1])
            else:
                cleaned.append(trajectory[i])
        
        return np.array(cleaned)
    
    @staticmethod
    def kalman_smooth(trajectory: np.ndarray, process_noise: float = 0.05) -> np.ndarray:
        """Kalman filter smoothing for trajectory"""
        
        if len(trajectory) < 2:
            return trajectory
        
        # Initialize Kalman filter
        kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State: [x, y, vx, vy]
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement function
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Initial state
        initial_velocity = trajectory[1] - trajectory[0] if len(trajectory) > 1 else np.array([0, 0])
        kf.x = np.array([trajectory[0, 0], trajectory[0, 1], 
                        initial_velocity[0], initial_velocity[1]])
        
        # Covariances
        kf.P *= 500
        kf.R = np.eye(2) * 10  # Measurement noise
        kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=process_noise, block_size=2)
        
        # Forward pass
        smoothed = []
        for point in trajectory:
            kf.predict()
            kf.update(point)
            smoothed.append(kf.x[:2].copy())
        
        return np.array(smoothed)
    
    @staticmethod
    def savitzky_golay_smooth(trajectory: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
        """Savitzky-Golay filter for smooth trajectories"""
        
        if len(trajectory) < window_length:
            return trajectory
        
        # Ensure window_length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Ensure we have enough points
        window_length = min(window_length, len(trajectory))
        if window_length < polyorder + 2:
            return trajectory
        
        smoothed = trajectory.copy()
        smoothed[:, 0] = savgol_filter(trajectory[:, 0], window_length, polyorder)
        smoothed[:, 1] = savgol_filter(trajectory[:, 1], window_length, polyorder)
        
        return smoothed
    
    @staticmethod
    def gaussian_smooth(trajectory: np.ndarray, sigma: float = 2.0) -> np.ndarray:
        """Gaussian smoothing"""
        
        if len(trajectory) < 3:
            return trajectory
        
        smoothed = trajectory.copy()
        smoothed[:, 0] = gaussian_filter1d(trajectory[:, 0], sigma=sigma)
        smoothed[:, 1] = gaussian_filter1d(trajectory[:, 1], sigma=sigma)
        
        return smoothed
    
    @staticmethod
    def ensemble_smooth(trajectory: np.ndarray) -> np.ndarray:
        """Combine multiple smoothing techniques"""
        
        if len(trajectory) < 3:
            return trajectory
        
        # Step 1: Remove outliers
        cleaned = TrajectorySmootherAdvanced.remove_outliers(trajectory, threshold=3.0)
        
        # Step 2: Kalman filter (handles noise well)
        kalman = TrajectorySmootherAdvanced.kalman_smooth(cleaned, process_noise=0.05)
        
        # Step 3: Savitzky-Golay (preserves features)
        if len(kalman) >= 7:
            savgol = TrajectorySmootherAdvanced.savitzky_golay_smooth(kalman, window_length=7, polyorder=2)
        else:
            savgol = kalman
        
        # Step 4: Light Gaussian (final smoothing)
        final = TrajectorySmootherAdvanced.gaussian_smooth(savgol, sigma=1.0)
        
        return final


class VehicleTrajectoryTracker:
    """
    Advanced trajectory tracker for drone videos
    """
    
    def __init__(
        self,
        video_path: str,
        confidence_threshold: float = 0.25,
        min_trajectory_length: int = 10,
        meters_per_pixel: Optional[float] = None,
        homography_matrix: Optional[np.ndarray] = None,
        homography_json: Optional[str] = None,
        roi_polygon: Optional[List[Tuple[int, int]]] = None
    ):
        # Resolve video path - check multiple locations
        video_path = self._resolve_video_path(video_path)
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold
        self.min_trajectory_length = min_trajectory_length
        self.meters_per_pixel = meters_per_pixel
        self.homography_matrix = homography_matrix
        self.roi_polygon = roi_polygon
        
        # Load homography from JSON if provided
        if homography_json and not homography_matrix:
            self.homography_matrix = self._load_homography_from_json(homography_json)
        
        # Initialize video analyzer with optimized SAHI settings
        self.analyzer = VehicleAnalyzer(
            model_conf=confidence_threshold,
            use_sahi=True,
            sahi_slice_size=1280  # Larger = faster, 1280 for 4K video
        )
        
        # Load detection model
        print("Loading VisDrone detection model...")
        self.analyzer.load_model()
        
        # Initialize tracker
        self.tracker = VehicleTracker(min_iou=0.25, max_age=30)
        
        # Storage for trajectories
        self.raw_trajectories: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.smooth_trajectories: Dict[int, np.ndarray] = {}
        self.vehicle_classes: Dict[int, str] = {}
        self.frame_indices: Dict[int, List[int]] = defaultdict(list)
        self.fps: Optional[float] = None
        self.total_frames: Optional[int] = None
        
        self.frame_width = 0
        self.frame_height = 0
    
    def _load_homography_from_json(self, json_path: str) -> np.ndarray:
        """Load homography matrix from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both direct matrix and nested structure
        if 'homography_matrix' in data:
            H = np.array(data['homography_matrix'], dtype=np.float32)
        else:
            # Assume the whole JSON is the matrix
            H = np.array(data, dtype=np.float32)
        
        print(f"  ✓ Loaded homography from: {json_path}")
        if 'points_used' in data:
            print(f"    Points used: {data['points_used']}")
        
        return H
    
    def _resolve_video_path(self, video_path: str) -> str:
        """Resolve video path from multiple possible locations"""
        from pathlib import Path
        
        # Check if it's an absolute path and exists
        if Path(video_path).exists():
            return str(Path(video_path).resolve())
        
        # Check in traffique_footage directory
        footage_dir = Path(r'C:\Users\sakth\Documents\traffique_footage')
        if (footage_dir / video_path).exists():
            return str((footage_dir / video_path).resolve())
        
        # Check in output directory
        output_dir = Path(__file__).parent / 'output'
        if (output_dir / video_path).exists():
            return str((output_dir / video_path).resolve())
        
        # Check in current directory
        if Path(video_path).exists():
            return str(Path(video_path).resolve())
        
        raise FileNotFoundError(
            f"Video not found: {video_path}\n"
            f"Checked:\n"
            f"  - {Path(video_path).resolve()}\n"
            f"  - {footage_dir / video_path}\n"
            f"  - {output_dir / video_path}"
        )
    
    def track_video(
        self,
        start_frame: int = 0,
        num_frames: int = 300
    ):
        """Track all vehicles in video and record trajectories"""
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        
        end_frame = self.total_frames if num_frames <= 0 else min(start_frame + num_frames, self.total_frames)
        frames_to_process = end_frame - start_frame
        
        print(f"\n╔══════════════════════════════════════════════════════════════════╗")
        print(f"║    ADVANCED VEHICLE TRAJECTORY TRACKER                          ║")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        print(f"\n📹 Video: {self.video_path}")
        print(f"📊 Frames: {start_frame} → {end_frame} ({frames_to_process} frames)")
        if self.fps:
            print(f"⚡ FPS: {self.fps:.1f}")
        else:
            print("⚡ FPS: unavailable (defaulting to raw frame counts)")
        print(f"🎯 Min trajectory length: {self.min_trajectory_length} frames")
        print(f"\n🧠 Noise Reduction Pipeline:")
        print(f"   1️⃣  Outlier Detection (MAD-based)")
        print(f"   2️⃣  Kalman Filter Smoothing")
        print(f"   3️⃣  Savitzky-Golay Filter")
        print(f"   4️⃣  Gaussian Smoothing")
        print("─" * 70)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Get frame dimensions
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_frame_polygon = [(0, 0), (self.frame_width, 0), 
                             (self.frame_width, self.frame_height), (0, self.frame_height)]
        
        frame_count = start_frame
        processed = 0
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Crop frame to ROI if specified (for speed)
            if self.roi_polygon and len(self.roi_polygon) == 4:
                # Simple rectangular crop
                y_min = min(p[1] for p in self.roi_polygon)
                y_max = max(p[1] for p in self.roi_polygon)
                x_min = min(p[0] for p in self.roi_polygon)
                x_max = max(p[0] for p in self.roi_polygon)
                
                # Crop and adjust ROI coordinates
                frame = frame[y_min:y_max, x_min:x_max]
                roi = [(0, 0), (x_max-x_min, 0), (x_max-x_min, y_max-y_min), (0, y_max-y_min)]
                
                # Store offset for coordinate correction
                offset_x = x_min
                offset_y = y_min
            else:
                offset_x = 0
                offset_y = 0
            
            # Detect vehicles
            result = self.analyzer.analyze_frame(frame, roi)
            detections = result['roi_vehicles']
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Store positions
            for vehicle_id, track_data in tracked_objects.items():
                bbox = track_data['bbox']
                center = np.array([
                    (bbox[0] + bbox[2]) / 2 + offset_x,  # Add offset back
                    (bbox[1] + bbox[3]) / 2 + offset_y   # Add offset back
                ])
                self.raw_trajectories[vehicle_id].append(center)
                self.frame_indices[vehicle_id].append(frame_count)
                
                # Store vehicle class
                if vehicle_id not in self.vehicle_classes:
                    self.vehicle_classes[vehicle_id] = track_data.get('class_name', 'vehicle')
            
            processed += 1
            if processed % 50 == 0:
                print(f"  ⏳ Processed {processed}/{frames_to_process} frames... "
                      f"({len(self.raw_trajectories)} vehicles tracked)")
            
            frame_count += 1
        
        cap.release()
        
        print(f"\n✅ Tracking complete!")
        print(f"   📍 Total vehicles detected: {len(self.raw_trajectories)}")
        
        # Smooth all trajectories
        self._smooth_all_trajectories()
        
        # Store the final frame for visualization
        self.final_frame_number = end_frame - 1
    
    def _smooth_all_trajectories(self):
        """Apply advanced smoothing to all trajectories"""
        
        print(f"\n🔧 Smoothing trajectories...")
        
        valid_count = 0
        
        for vehicle_id, raw_positions in self.raw_trajectories.items():
            if len(raw_positions) < self.min_trajectory_length:
                continue
            
            raw_array = np.array(raw_positions)
            
            # Apply ensemble smoothing
            smoothed = TrajectorySmootherAdvanced.ensemble_smooth(raw_array)
            
            self.smooth_trajectories[vehicle_id] = smoothed
            valid_count += 1
        
        print(f"   ✅ Smoothed {valid_count} valid trajectories")
        print(f"   ❌ Filtered out {len(self.raw_trajectories) - valid_count} short tracks")
    
    def visualize_all_trajectories(
        self,
        output_dir: str = "output/trajectories"
    ):
        """Visualize all trajectories on final frame"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the final frame
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.final_frame_number)
        ret, final_frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to get final frame")
            return
        
        print(f"\n🎨 Creating visualizations...")
        print("─" * 70)
        
        # Color palette for different vehicles
        color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 255, 0), (0, 128, 255),
            (255, 0, 128), (128, 0, 255), (0, 255, 128),
            (192, 64, 0), (64, 192, 0), (0, 64, 192),
            (192, 0, 64), (64, 0, 192), (0, 192, 64)
        ]
        
        # Create visualization
        vis_frame = final_frame.copy()
        
        for idx, (vehicle_id, trajectory) in enumerate(self.smooth_trajectories.items()):
            color = color_palette[idx % len(color_palette)]
            
            # Draw trajectory line with gradient effect
            num_points = len(trajectory)
            for i in range(1, num_points):
                # Gradient from darker to brighter
                alpha = 0.3 + 0.7 * (i / num_points)
                current_color = tuple(int(c * alpha) for c in color)
                
                # Line thickness increases towards end
                thickness = max(1, int(1 + 2 * (i / num_points)))
                
                pt1 = tuple(trajectory[i-1].astype(int))
                pt2 = tuple(trajectory[i].astype(int))
                cv2.line(vis_frame, pt1, pt2, current_color, thickness, cv2.LINE_AA)
            
            # Draw start point (green circle)
            start_pt = tuple(trajectory[0].astype(int))
            cv2.circle(vis_frame, start_pt, 5, (0, 255, 0), -1)
            cv2.circle(vis_frame, start_pt, 7, (255, 255, 255), 2)
            
            # Draw end point (red circle - current vehicle position)
            end_pt = tuple(trajectory[-1].astype(int))
            cv2.circle(vis_frame, end_pt, 7, (0, 0, 255), -1)
            cv2.circle(vis_frame, end_pt, 9, (255, 255, 255), 2)
            
            # Draw vehicle ID near end point
            label_pos = (end_pt[0] + 18, end_pt[1] - 8)
            cv2.putText(vis_frame, f"ID:{vehicle_id}", label_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        
        # Add info panel
        panel_height = 120
        panel = np.zeros((panel_height, vis_frame.shape[1], 3), dtype=np.uint8)
        
        info_lines = [
            f"╔══════════════════════════════════════════════════════════════════╗",
            f"║  VEHICLE TRAJECTORIES - FINAL FRAME                              ║",
            f"╠══════════════════════════════════════════════════════════════════╣",
            f"║  Total Vehicles: {len(self.smooth_trajectories):<4}  |  Legend:  🟢 Start  🔴 End (Current)    ║",
            f"║  Frame: {self.final_frame_number:<6}       |  Smoothing: Multi-Algorithm         ║",
            f"╚══════════════════════════════════════════════════════════════════╝"
        ]
        
        y_offset = 15
        for line in info_lines:
            cv2.putText(panel, line, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 20
        
        # Combine
        result = np.vstack([panel, vis_frame])
        
        # Save
        output_file = output_path / "all_trajectories.png"
        cv2.imwrite(str(output_file), result)
        print(f"  ✅ Saved: {output_file.name}")
        
        return str(output_file)
    
    def visualize_individual_trajectories(
        self,
        output_dir: str = "output/trajectories",
        num_vehicles: int = 5
    ):
        """Create individual visualizations for top vehicles"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get the final frame
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.final_frame_number)
        ret, final_frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to get final frame")
            return
        
        # Sort vehicles by trajectory length
        sorted_vehicles = sorted(
            self.smooth_trajectories.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )[:num_vehicles]
        
        print(f"\n🎨 Creating {len(sorted_vehicles)} individual trajectory visualizations...")
        print("─" * 70)
        
        for idx, (vehicle_id, trajectory) in enumerate(sorted_vehicles, 1):
            # Create fresh frame
            img = final_frame.copy()
            
            # Highlight color for this vehicle
            main_color = (0, 255, 255)  # Cyan
            
            # Draw trajectory with beautiful gradient
            num_points = len(trajectory)
            for i in range(1, num_points):
                # Color gradient from blue to cyan
                ratio = i / num_points
                color = (
                    int(255 * ratio),
                    int(100 + 155 * ratio),
                    int(255)
                )
                
                thickness = max(2, int(2 + 4 * ratio))
                
                pt1 = tuple(trajectory[i-1].astype(int))
                pt2 = tuple(trajectory[i].astype(int))
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
            
            # Draw waypoints along trajectory (every 10th point)
            for i in range(0, num_points, 10):
                pt = tuple(trajectory[i].astype(int))
                alpha = i / num_points
                size = int(3 + 2 * alpha)
                cv2.circle(img, pt, size, (255, 255, 255), -1)
                cv2.circle(img, pt, size + 2, (0, 0, 0), 1)
            
            # Draw start point (large green)
            start_pt = tuple(trajectory[0].astype(int))
            cv2.circle(img, start_pt, 10, (0, 255, 0), -1)
            cv2.circle(img, start_pt, 12, (255, 255, 255), 2)
            cv2.putText(img, "START", (start_pt[0] - 30, start_pt[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            
            # Draw end point (large red - current position)
            end_pt = tuple(trajectory[-1].astype(int))
            cv2.circle(img, end_pt, 12, (0, 0, 255), -1)
            cv2.circle(img, end_pt, 14, (255, 255, 255), 2)
            cv2.putText(img, "END", (end_pt[0] + 25, end_pt[1] - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Calculate trajectory statistics
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            total_distance = np.sum(distances)
            avg_speed = np.mean(distances)
            
            # Calculate displacement
            displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
            
            # Info panel
            panel_height = 180
            panel = np.zeros((panel_height, img.shape[1], 3), dtype=np.uint8)
            
            vehicle_class = self.vehicle_classes.get(vehicle_id, 'vehicle')
            
            info_lines = [
                f"╔══════════════════════════════════════════════════════════════════╗",
                f"║  VEHICLE #{idx} - ID: {vehicle_id:<5}                                        ║",
                f"╠══════════════════════════════════════════════════════════════════╣",
                f"║  Type: {vehicle_class:<15}                                        ║",
                f"║  Trajectory Length: {num_points} frames                              ║",
                f"║  Total Distance: {total_distance:.1f} pixels                            ║",
                f"║  Displacement: {displacement:.1f} pixels                              ║",
                f"║  Avg Speed: {avg_speed:.2f} px/frame                                  ║",
                f"╚══════════════════════════════════════════════════════════════════╝"
            ]
            
            y_offset = 15
            for line in info_lines:
                cv2.putText(panel, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 20
            
            # Combine
            result = np.vstack([panel, img])
            
            # Save
            output_file = output_path / f"trajectory_{idx:02d}_vehicle_{vehicle_id}.png"
            cv2.imwrite(str(output_file), result)
            print(f"  ✅ Vehicle {idx}: {output_file.name}")
            print(f"      • Frames: {num_points} | Distance: {total_distance:.1f}px | Type: {vehicle_class}")
        
        print("─" * 70)

    def export_trajectories_csv(
        self,
        output_csv: str = "output/trajectories/trajectoriesd2f1.csv",
        format_style: str = "default"
    ) -> str:
        """
        Export smoothed trajectories to CSV with optional meter scaling
        
        Args:
            output_csv: Output CSV path
            format_style: "default" or "d2f1" (matches D2F1_lclF.csv format)
        """

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.smooth_trajectories:
            print("No trajectories to export.")
            return str(output_path)

        frame_time = 1.0 / self.fps if self.fps and self.fps > 0 else None
        use_meters = self.meters_per_pixel is not None and self.meters_per_pixel > 0

        rows = []
        for vehicle_id, trajectory in self.smooth_trajectories.items():
            frames = self.frame_indices.get(vehicle_id, [])
            vehicle_class = self.vehicle_classes.get(vehicle_id, 'Car')
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1) if len(trajectory) > 1 else np.array([])
            speeds_px_per_frame = np.concatenate([[0.0], distances]) if len(trajectory) > 1 else np.array([0.0])

            for idx, point in enumerate(trajectory):
                frame_no = frames[idx] if idx < len(frames) else idx
                time_seconds = frame_no * frame_time if frame_time is not None else None
                speed_px_frame = float(speeds_px_per_frame[idx]) if idx < len(speeds_px_per_frame) else 0.0
                speed_px_second = speed_px_frame * self.fps if self.fps else None

                if format_style == "d2f1":
                    # Match D2F1_lclF.csv format: Frame, Class, VehicleID, X_pixel, Y_pixel, Time, X_world, Y_world
                    
                    # Convert pixel → world coordinates
                    if self.homography_matrix is not None:
                        # Use homography transformation
                        point_h = np.array([float(point[0]), float(point[1]), 1.0])
                        world_h = self.homography_matrix @ point_h
                        x_world = float(world_h[0] / world_h[2])
                        y_world = float(world_h[1] / world_h[2])
                    elif use_meters:
                        # Simple scale factor
                        x_world = float(point[0]) * self.meters_per_pixel
                        y_world = float(point[1]) * self.meters_per_pixel
                    else:
                        x_world = 0.0
                        y_world = 0.0
                    
                    row = {
                        "Frame": frame_no,
                        "Class": vehicle_class.capitalize(),
                        "VehicleID": f"{vehicle_class}_{vehicle_id}",
                        "X_pixel": round(float(point[0]), 4),
                        "Y_pixel": round(float(point[1]), 4),
                        "Time": round(time_seconds, 2) if time_seconds is not None else 0.0,
                        "X_world": round(x_world, 6),
                        "Y_world": round(y_world, 6)
                    }
                else:
                    # Default format
                    row = {
                        "vehicle_id": vehicle_id,
                        "frame": frame_no,
                        "time_seconds": round(time_seconds, 4) if time_seconds is not None else "",
                        "x_px": round(float(point[0]), 2),
                        "y_px": round(float(point[1]), 2),
                        "speed_px_per_frame": round(speed_px_frame, 4),
                        "speed_px_per_second": round(speed_px_second, 4) if speed_px_second is not None else ""
                    }

                    if use_meters:
                        x_m = float(point[0]) * self.meters_per_pixel
                        y_m = float(point[1]) * self.meters_per_pixel
                        speed_m_second = speed_px_second * self.meters_per_pixel if speed_px_second is not None else None
                        row.update({
                            "x_m": round(x_m, 3),
                            "y_m": round(y_m, 3),
                            "speed_m_per_second": round(speed_m_second, 4) if speed_m_second is not None else ""
                        })

                rows.append(row)

        fieldnames = list(rows[0].keys()) if rows else []

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"  📄 Saved trajectory CSV ({format_style} format): {output_path}")
        return str(output_path)
    
    def get_statistics(self) -> Dict:
        """Get trajectory statistics"""
        
        stats = {
            'total_vehicles': len(self.smooth_trajectories),
            'trajectories': {}
        }
        
        for vehicle_id, trajectory in self.smooth_trajectories.items():
            distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
            
            stats['trajectories'][vehicle_id] = {
                'length': len(trajectory),
                'total_distance': float(np.sum(distances)),
                'avg_speed': float(np.mean(distances)),
                'displacement': float(np.linalg.norm(trajectory[-1] - trajectory[0])),
                'start': trajectory[0].tolist(),
                'end': trajectory[-1].tolist(),
                'class': self.vehicle_classes.get(vehicle_id, 'vehicle')
            }
        
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Vehicle Trajectory Tracker for Drone Videos"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--start", type=int, default=9700, help="Start frame")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to process (-1 for full video)")
    parser.add_argument("--output", default="output/trajectories", help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of individual trajectories to visualize")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum trajectory length")
    parser.add_argument("--meters-per-pixel", type=float, default=None,
                        help="Scale to convert pixel distances to meters (meters per pixel)")
    parser.add_argument("--csv", default="output/trajectories/trajectories.csv",
                        help="Path to save trajectory CSV output")
    parser.add_argument("--format", choices=["default", "d2f1"], default="default",
                        help="CSV format: 'default' or 'd2f1' (matches D2F1_lclF.csv)")
    parser.add_argument("--homography", type=str, default=None,
                        help="Path to homography JSON file for pixel→world coordinate transformation")
    parser.add_argument("--roi", type=str, default=None,
                        help="ROI region: 'road' for auto road detection (Y: 850-1100) or custom JSON array")
    
    args = parser.parse_args()
    
    # Parse ROI if provided
    roi_polygon = None
    if args.roi:
        if args.roi.lower() == 'road':
            # Auto ROI for road region (Y: 850-1100 based on Sohaib's data)
            roi_polygon = [(0, 850), (3840, 850), (3840, 1100), (0, 1100)]
            print("🛣️  Using auto road ROI: Y=850-1100")
        else:
            try:
                import json
                coords = json.loads(args.roi)
                roi_polygon = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
                print(f"🎯 Using custom ROI: {len(roi_polygon)} points")
            except:
                print("⚠️  Invalid ROI format, using full frame")
                roi_polygon = None
    
    print("\n" + "="*70)
    print("  ADVANCED VEHICLE TRAJECTORY TRACKER")
    print("  Multi-Algorithm Noise Reduction for Drone Videos")
    print("="*70 + "\n")
    
    # Create tracker
    tracker = VehicleTrajectoryTracker(
        video_path=args.video,
        confidence_threshold=args.confidence,
        min_trajectory_length=args.min_length,
        meters_per_pixel=args.meters_per_pixel,
        homography_json=args.homography,
        roi_polygon=roi_polygon
    )
    
    # Track video
    tracker.track_video(
        start_frame=args.start,
        num_frames=args.frames
    )
    
    # Visualize all trajectories
    tracker.visualize_all_trajectories(output_dir=args.output)
    
    # Visualize individual trajectories
    tracker.visualize_individual_trajectories(
        output_dir=args.output,
        num_vehicles=args.top_k
    )

    # Export CSV
    csv_path = tracker.export_trajectories_csv(output_csv=args.csv, format_style=args.format)
    
    # Print statistics
    stats = tracker.get_statistics()
    print(f"\n📊 SUMMARY:")
    print(f"   • Total vehicles tracked: {stats['total_vehicles']}")
    print(f"   • CSV saved to: {csv_path}")
    if tracker.homography_matrix is not None:
        print(f"   • Homography transformation applied ✅")
    elif args.meters_per_pixel:
        print(f"   • Meter scale applied: {args.meters_per_pixel} m/px")
    
    # Show top 3 by distance
    sorted_by_distance = sorted(
        stats['trajectories'].items(),
        key=lambda x: x[1]['total_distance'],
        reverse=True
    )[:3]
    
    print(f"\n   🏆 Top 3 by distance traveled:")
    for idx, (vid, data) in enumerate(sorted_by_distance, 1):
        if args.meters_per_pixel:
            distance_m = data['total_distance'] * args.meters_per_pixel
            print(f"      {idx}. Vehicle {vid}: {data['total_distance']:.1f} px ({distance_m:.2f} m) ({data['length']} frames)")
        else:
            print(f"      {idx}. Vehicle {vid}: {data['total_distance']:.1f} pixels ({data['length']} frames)")
    
    print("\n" + "="*70)
    print("  ✅ COMPLETE! Check the output directory for visualizations.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
