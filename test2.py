#!/usr/bin/env python3
"""
Advanced Trajectory Prediction Integration
Integrates prediction with video analysis pipeline

Features:
- Extracts vehicle tracklets from video
- Predicts smooth, multi-modal future paths
- Visualizes predictions with uncertainty-style modes

This version uses a robust kinematic + smoothing-based algorithm:
- Smooths history with Kalman and moving average
- Estimates average recent velocity
- Extrapolates until vehicle exits the frame
- Creates 3 smooth modes: straight, slight left, slight right
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
from collections import defaultdict

# Import existing modules
from interactive_analytics import VehicleAnalyzer, VehicleTracker
from utils.trajectory import smooth_kalman
from trajectory_prediction_system import TrajectoryPredictor


class AdvancedTrajectoryAnalyzer:
    """
    Integrates prediction model with video analysis
    """
    
    def __init__(
        self,
        video_path: str,
        model_path: str = None,
        confidence_threshold: float = 0.25,
        history_length: int = 20,
        future_length: int = 30
    ):
        self.video_path = video_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize video analyzer
        self.analyzer = VehicleAnalyzer(
            model_conf=confidence_threshold,
            use_sahi=True,
            sahi_slice_size=640
        )
        
        # Load detection model
        print("Loading VisDrone detection model...")
        self.analyzer.load_model()
        
        # Initialize tracker
        self.tracker = VehicleTracker(min_iou=0.25, max_age=25)
        
        # Predictor used for config (history_length, future_length, num_modes)
        self.predictor = TrajectoryPredictor(
            model_path=model_path,
            history_length=history_length,
            future_length=future_length,
            num_modes=3
        )
        
        # Storage for trajectories
        self.trajectories: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.predictions: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self.frame_width = 0
        self.frame_height = 0

    # ---------- SMOOTHING & GEOMETRY HELPERS ----------

    @staticmethod
    def _moving_average(points: np.ndarray, window: int = 5) -> np.ndarray:
        """
        Simple moving-average smoother for a sequence of 2D points.
        points: (T, 2)
        """
        if len(points) <= 2 or window <= 1:
            return points

        window = min(window, len(points))
        half = window // 2
        smoothed = points.copy()

        for i in range(len(points)):
            start = max(0, i - half)
            end = min(len(points), i + half + 1)
            smoothed[i] = points[start:end].mean(axis=0)

        return smoothed

    @staticmethod
    def _rotate_vector(v: np.ndarray, angle_rad: float) -> np.ndarray:
        """
        Rotate 2D vector v by angle_rad.
        """
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        R = np.array([[c, -s],
                      [s,  c]])
        return R @ v

    def _predict_until_exit(self, last_position: np.ndarray, velocity: np.ndarray) -> int:
        """Calculate an upper bound on how many frames until vehicle exits frame."""
        margin = 50
        
        x, y = last_position
        vx, vy = velocity
        
        # If almost no motion, just predict a short horizon
        speed = np.linalg.norm(velocity)
        if speed < 0.5:
            return min(self.predictor.future_length, 60)
        
        frames_to_boundaries = []
        
        if vx > 0:  # Moving right
            frames_to_boundaries.append((self.frame_width + margin - x) / vx)
        elif vx < 0:  # Moving left
            frames_to_boundaries.append((x + margin) / abs(vx))
        
        if vy > 0:  # Moving down
            frames_to_boundaries.append((self.frame_height + margin - y) / vy)
        elif vy < 0:  # Moving up
            frames_to_boundaries.append((y + margin) / abs(vy))
        
        valid_frames = [f for f in frames_to_boundaries if f > 0]
        if valid_frames:
            return int(min(min(valid_frames), self.predictor.future_length * 2, 300))
        
        # Default: limited horizon
        return min(self.predictor.future_length, 60)
    
    # ---------- MAIN VIDEO ANALYSIS ----------

    def analyze_video(
        self,
        start_frame: int = 0,
        end_frame: int = None,
        max_frames: int = 500
    ):
        """
        Analyze video and build trajectories
        
        Args:
            start_frame: Starting frame
            end_frame: Ending frame (None = until end or max_frames)
            max_frames: Maximum frames to process
        """
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if end_frame is None:
            end_frame = min(start_frame + max_frames, total_frames)
        else:
            end_frame = min(end_frame, total_frames)
        
        print(f"\nADVANCED TRAJECTORY PREDICTION")
        print("="*70)
        print(f"Video: {self.video_path}")
        print(f"Frames: {start_frame} to {end_frame} ({end_frame - start_frame} frames)")
        print(f"FPS: {fps}")
        print(f"History: {self.predictor.history_length} frames")
        print(f"Prediction: up to {self.predictor.future_length} frames (adaptive until exit)")
        print("="*70)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        processed = 0
        
        # Get frame dimensions for full ROI
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_frame_polygon = [
            (0, 0),
            (self.frame_width, 0),
            (self.frame_width, self.frame_height),
            (0, self.frame_height)
        ]
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles (using full frame as ROI)
            result = self.analyzer.analyze_frame(frame, full_frame_polygon)
            detections = result['roi_vehicles']
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Store positions (tracked_objects is a dict: {id: {...}})
            for vehicle_id, track_data in tracked_objects.items():
                bbox = track_data['bbox']
                center = np.array([
                    (bbox[0] + bbox[2]) / 2.0,
                    (bbox[1] + bbox[3]) / 2.0
                ], dtype=np.float32)
                self.trajectories[vehicle_id].append(center)
            
            processed += 1
            if processed % 50 == 0:
                print(f"  Processed {processed} frames... ({len(self.trajectories)} vehicles tracked)")
            
            frame_count += 1
        
        cap.release()
        
        print(f"\n✓ Analysis complete!")
        print(f"  Total vehicles tracked: {len(self.trajectories)}")
        
        # Generate predictions for vehicles with sufficient history
        self._generate_predictions()
    
    # ---------- PREDICTION LOGIC (KINEMATIC + SMOOTHING) ----------

    def _generate_predictions(self):
        """
        Generate smooth predictions for all vehicles with sufficient history.

        Algorithm (per vehicle):
        1. Take last N history points (N = history_length).
        2. Smooth with Kalman + moving average.
        3. Estimate average recent velocity.
        4. Predict until vehicle exits the frame or max horizon is reached:
           - Mode 1: straight (main mode)
           - Mode 2: slight left rotation of velocity
           - Mode 3: slight right rotation of velocity
        5. Smooth each future trajectory and clamp to frame.
        """
        print(f"\nGenerating predictions (kinematic + smoothing)...")

        min_history = max(5, self.predictor.history_length)  # need enough points for stable velocity
        predicted_count = 0
        
        for vehicle_id, positions in self.trajectories.items():
            if len(positions) < min_history:
                continue
            
            history = np.array(positions, dtype=np.float32)
            # Use the last min_history points
            recent = history[-min_history:]
            
            # Kalman smoothing (user-provided util)
            smoothed = smooth_kalman(recent)  # (T, 2)
            # Additional moving average for super smooth history
            smoothed = self._moving_average(smoothed, window=5)
            
            # Estimate velocity from last few steps
            diffs = np.diff(smoothed, axis=0)  # (T-1, 2)
            if len(diffs) == 0:
                continue
            
            tail_len = min(5, len(diffs))
            avg_velocity = diffs[-tail_len:].mean(axis=0)  # (2,)
            speed = np.linalg.norm(avg_velocity)
            
            last_pos = smoothed[-1].copy()
            
            # If almost stationary, create short, small prediction around last_pos
            if speed < 0.3:
                future_len = self.predictor.future_length
                base_traj = np.tile(last_pos[None, :], (future_len, 1))
                
                # Add very slight drift to avoid perfectly flat line
                base_traj[:, 0] += np.linspace(0, 3, future_len)
                
                # All modes almost identical
                future_modes = np.stack([base_traj, base_traj, base_traj], axis=0)
                mode_probs = np.array([0.7, 0.2, 0.1], dtype=np.float32)
                self.predictions[vehicle_id] = (future_modes, mode_probs)
                predicted_count += 1
                continue
            
            # Determine horizon based on frame exit
            max_frames_until_exit = self._predict_until_exit(last_pos, avg_velocity)
            max_horizon = min(max_frames_until_exit, self.predictor.future_length * 2)
            if max_horizon <= 0:
                continue
            
            # Define 3 velocity directions: straight, left-rotated, right-rotated
            straight_v = avg_velocity
            angle = np.deg2rad(6.0)  # slight curvature
            left_v = self._rotate_vector(straight_v, angle)
            right_v = self._rotate_vector(straight_v, -angle)
            
            velocities = [straight_v, left_v, right_v]
            mode_probs = np.array([0.6, 0.25, 0.15], dtype=np.float32)
            
            mode_trajectories = []
            for v in velocities:
                traj_points = []
                current_pos = last_pos.copy()
                
                for _ in range(max_horizon):
                    current_pos = current_pos + v  # constant velocity step
                    
                    # Stop if out of frame
                    x, y = current_pos
                    if x < -10 or x > self.frame_width + 10 or y < -10 or y > self.frame_height + 10:
                        break
                    
                    # Clamp within frame bounds for visualization
                    clamped = np.array([
                        np.clip(current_pos[0], 0, self.frame_width - 1),
                        np.clip(current_pos[1], 0, self.frame_height - 1)
                    ], dtype=np.float32)
                    traj_points.append(clamped)
                
                if len(traj_points) == 0:
                    # If something went wrong, keep at least one point
                    traj_points = [last_pos.copy()]
                
                traj = np.vstack(traj_points)  # (T_f, 2)
                # Smooth predicted trajectory with moving average
                traj = self._moving_average(traj, window=7)
                mode_trajectories.append(traj)
            
            # Pad / truncate all modes to same length for consistency
            max_len = max(len(t) for t in mode_trajectories)
            aligned_modes = []
            for traj in mode_trajectories:
                if len(traj) < max_len:
                    # Repeat last point to pad
                    pad = np.tile(traj[-1], (max_len - len(traj), 1))
                    traj = np.vstack([traj, pad])
                aligned_modes.append(traj)
            
            future_modes = np.stack(aligned_modes, axis=0)  # (3, T_f, 2)
            
            # Normalize probabilities just in case
            mode_probs = mode_probs / mode_probs.sum()
            
            # Debug print for first few vehicles
            if predicted_count < 3:
                print(
                    f"  Vehicle {vehicle_id}: "
                    f"history={len(history)} frames, "
                    f"used={len(recent)} frames, "
                    f"future={future_modes.shape[1]} frames"
                )
            
            self.predictions[vehicle_id] = (future_modes, mode_probs)
            predicted_count += 1
        
        print(f"  ✓ Generated smooth predictions for {predicted_count} vehicles")
    
    # ---------- NEIGHBOR EXTRACTION (OPTIONAL, CURRENTLY UNUSED IN PREDICTION) ----------

    def _get_neighbor_trajectories(
        self,
        ego_id: int,
        ego_positions: np.ndarray,
        max_neighbors: int = 5
    ) -> List[np.ndarray]:
        """
        Get trajectories of nearby vehicles (not used in current prediction algorithm,
        but kept for future model-based extension).
        """
        neighbors = []
        ego_last_pos = ego_positions[-1]
        
        distances = []
        for vehicle_id, positions in self.trajectories.items():
            if vehicle_id == ego_id:
                continue
            
            if len(positions) >= len(ego_positions):
                aligned_positions = np.array(positions[-len(ego_positions):], dtype=np.float32)
                last_pos = aligned_positions[-1]
                dist = np.linalg.norm(last_pos - ego_last_pos)
                distances.append((dist, vehicle_id, aligned_positions))
        
        distances.sort(key=lambda x: x[0])
        
        for _, _, positions in distances[:max_neighbors]:
            neighbors.append(positions)
        
        return neighbors
    
    # ---------- VISUALIZATION ----------

    def visualize_predictions(
        self,
        output_dir: str = "output/predictions",
        top_k: int = 5
    ):
        """
        Visualize predicted trajectories
        
        Args:
            output_dir: Output directory
            top_k: Number of top vehicles to visualize
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open video to get background frame
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Get middle frame as background
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, background = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to get background frame")
            return
        
        # Sort vehicles by trajectory length
        sorted_vehicles = sorted(
            self.predictions.items(),
            key=lambda x: len(self.trajectories[x[0]]),
            reverse=True
        )[:top_k]
        
        print(f"\nVisualizing top {len(sorted_vehicles)} predictions...")
        
        for idx, (vehicle_id, (future_modes, mode_probs)) in enumerate(sorted_vehicles, 1):
            img = background.copy()
            
            # Get historical trajectory
            history = np.array(self.trajectories[vehicle_id], dtype=np.float32)
            
            # Draw history (green)
            for i in range(1, len(history)):
                pt1 = tuple(history[i-1].astype(int))
                pt2 = tuple(history[i].astype(int))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            
            # Draw last position
            last_pos = tuple(history[-1].astype(int))
            cv2.circle(img, last_pos, 8, (0, 255, 0), -1)
            
            # Draw all predicted modes
            colors = [
                (255, 0, 0),    # Blue - Mode 1
                (0, 165, 255),  # Orange - Mode 2
                (255, 255, 0)   # Cyan - Mode 3
            ]
            
            for mode_idx, (mode_traj, prob) in enumerate(zip(future_modes, mode_probs)):
                color = colors[mode_idx % len(colors)]
                mode_traj = mode_traj.astype(int)
                
                # Connect to history
                cv2.line(img, last_pos, tuple(mode_traj[0]), color, 3)
                
                # Draw entire predicted path (smooth AA line)
                for i in range(1, len(mode_traj)):
                    pt1 = tuple(mode_traj[i-1])
                    pt2 = tuple(mode_traj[i])
                    cv2.line(img, pt1, pt2, color, 2, cv2.LINE_AA)
                
                # Draw endpoint with larger circle
                endpoint = tuple(mode_traj[-1])
                cv2.circle(img, endpoint, 10, color, -1)
                cv2.circle(img, endpoint, 12, (255, 255, 255), 2)  # White outline
                
                # Add probability label near endpoint
                label_pos = (endpoint[0] + 15, endpoint[1] + 5)
                cv2.putText(
                    img,
                    f"{prob:.0%}",
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA
                )
            
            # Add info panel
            panel_height = 180
            panel = np.zeros((panel_height, img.shape[1], 3), dtype=np.uint8)
            
            info_lines = [
                f"Vehicle ID: {vehicle_id}",
                f"History Length: {len(history)} frames",
                f"Prediction Horizon: {future_modes.shape[1]} frames",
                "",
                "Predicted Modes:",
                f"  Mode 1 (Blue):   {mode_probs[0]:.1%}" if len(mode_probs) > 0 else "",
                f"  Mode 2 (Orange): {mode_probs[1]:.1%}" if len(mode_probs) > 1 else "",
                f"  Mode 3 (Cyan):   {mode_probs[2]:.1%}" if len(mode_probs) > 2 else ""
            ]
            
            for i, line in enumerate(info_lines):
                if not line:
                    continue
                cv2.putText(
                    panel,
                    line,
                    (10, 25 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
            
            # Combine
            result = np.vstack([img, panel])
            
            # Save
            output_file = output_path / f"prediction_{idx}_vehicle_{vehicle_id}.png"
            cv2.imwrite(str(output_file), result)
            print(f"  Saved: {output_file}")
        
        # Create all-vehicles visualization
        print(f"\nCreating all-vehicles overview...")
        img_all = background.copy()
        
        # Use different colors for different vehicles
        color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 255), (255, 128, 0), (0, 255, 128),
            (255, 0, 128), (128, 255, 0), (0, 128, 255)
        ]
        
        for vehicle_idx, (vehicle_id, (future_modes, mode_probs)) in enumerate(self.predictions.items()):
            color = color_palette[vehicle_idx % len(color_palette)]
            
            # Get historical trajectory
            history = np.array(self.trajectories[vehicle_id], dtype=np.float32)
            
            # Draw history (thinner, semi-transparent effect with darker color)
            history_color = tuple(int(c * 0.6) for c in color)
            for i in range(1, len(history)):
                pt1 = tuple(history[i-1].astype(int))
                pt2 = tuple(history[i].astype(int))
                cv2.line(img_all, pt1, pt2, history_color, 1)
            
            # Draw predicted path (main mode only, thicker)
            predicted = future_modes[0].astype(int)
            last_pos = tuple(history[-1].astype(int))
            
            if len(predicted) > 0:
                cv2.line(img_all, last_pos, tuple(predicted[0]), color, 2)
                for i in range(1, len(predicted)):
                    pt1 = tuple(predicted[i-1])
                    pt2 = tuple(predicted[i])
                    cv2.line(img_all, pt1, pt2, color, 2, cv2.LINE_AA)
                
                # Mark endpoint
                endpoint = tuple(predicted[-1])
                cv2.circle(img_all, endpoint, 6, color, -1)
                cv2.circle(img_all, endpoint, 8, (255, 255, 255), 1)
            
            # Mark last observed position
            cv2.circle(img_all, last_pos, 5, color, -1)
        
        # Add title
        title_panel = np.zeros((60, img_all.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            title_panel,
            f"All Vehicle Trajectory Predictions ({len(self.predictions)} vehicles)",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        
        result_all = np.vstack([title_panel, img_all])
        
        # Save all-vehicles visualization
        output_file_all = output_path / "all_vehicles_predictions.png"
        cv2.imwrite(str(output_file_all), result_all)
        print(f"  Saved all-vehicles: {output_file_all}")
        
        print(f"\n✓ Visualizations saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Advanced Trajectory Prediction")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", help="Path to pretrained model (optional)")
    parser.add_argument("--start", type=int, default=9700, help="Start frame")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames to process")
    parser.add_argument("--output", default="output/predictions", help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of vehicles to visualize")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    analyzer = AdvancedTrajectoryAnalyzer(
        video_path=args.video,
        model_path=args.model,
        confidence_threshold=args.confidence,
        history_length=20,
        future_length=30
    )
    
    analyzer.analyze_video(
        start_frame=args.start,
        max_frames=args.frames
    )
    
    analyzer.visualize_predictions(
        output_dir=args.output,
        top_k=args.top_k
    )
    
    print("\n" + "="*70)
    print("Advanced trajectory prediction complete!")
    print("="*70)


if __name__ == "__main__":
    main()
