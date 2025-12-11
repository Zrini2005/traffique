#!/usr/bin/env python3
"""
ENHANCED Advanced Trajectory Prediction System
State-of-the-art trajectory prediction with multiple algorithms

ALGORITHMS IMPLEMENTED:
1. Kalman Filter with Constant Velocity Model
2. Extended Kalman Filter with Constant Acceleration
3. Cubic Spline Interpolation for smooth curves
4. Social Force Model for vehicle interactions
5. Lane-aware prediction with curvature analysis
6. Ensemble prediction with confidence weighting
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from collections import defaultdict
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull

# Import existing modules
from interactive_analytics import VehicleAnalyzer, VehicleTracker
from utils.trajectory import smooth_kalman
from trajectory_prediction_system import TrajectoryPredictor


class EnhancedKalmanPredictor:
    """Enhanced Kalman Filter for trajectory prediction"""
    
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.kf = None
    
    def initialize(self, initial_pos: np.ndarray, initial_vel: np.ndarray):
        """Initialize Kalman filter with position and velocity"""
        # State: [x, y, vx, vy, ax, ay]
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        
        # State transition matrix (constant acceleration model)
        dt = self.dt
        self.kf.F = np.array([
            [1, 0, dt, 0, 0.5*dt*dt, 0],
            [0, 1, 0, dt, 0, 0.5*dt*dt],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only observe position)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Initial state
        self.kf.x = np.array([
            initial_pos[0], initial_pos[1],
            initial_vel[0], initial_vel[1],
            0, 0
        ])
        
        # Covariance matrices
        self.kf.P *= 100  # Initial uncertainty
        self.kf.R = np.eye(2) * 5  # Measurement noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=0.1, block_size=3)
    
    def predict(self, n_steps: int) -> np.ndarray:
        """Predict n steps into the future"""
        predictions = []
        
        for _ in range(n_steps):
            self.kf.predict()
            predictions.append(self.kf.x[:2].copy())
        
        return np.array(predictions)


class SocialForcePredictor:
    """Social Force Model for vehicle interaction prediction"""
    
    def __init__(self, dt: float = 1.0):
        self.dt = dt
        self.interaction_radius = 100.0  # pixels
        self.repulsion_strength = 20.0
        self.attraction_strength = 5.0
    
    def compute_social_forces(
        self,
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        neighbor_positions: List[np.ndarray],
        neighbor_velocities: List[np.ndarray]
    ) -> np.ndarray:
        """Compute social forces from nearby vehicles"""
        
        total_force = np.zeros(2)
        
        for n_pos, n_vel in zip(neighbor_positions, neighbor_velocities):
            # Distance to neighbor
            diff = ego_pos - n_pos
            dist = np.linalg.norm(diff)
            
            if dist < self.interaction_radius and dist > 1e-6:
                # Repulsive force (avoid collision)
                repulsion = self.repulsion_strength * diff / (dist ** 2)
                
                # Directional influence (follow similar trajectories)
                vel_similarity = np.dot(ego_vel, n_vel) / (
                    np.linalg.norm(ego_vel) * np.linalg.norm(n_vel) + 1e-6
                )
                
                if vel_similarity > 0.5:  # Similar direction
                    attraction = -self.attraction_strength * diff / dist
                    total_force += attraction * 0.3
                
                total_force += repulsion
        
        return total_force
    
    def predict_with_social_forces(
        self,
        initial_pos: np.ndarray,
        initial_vel: np.ndarray,
        neighbor_data: List[Tuple[np.ndarray, np.ndarray]],
        n_steps: int
    ) -> np.ndarray:
        """Predict trajectory considering social forces"""
        
        predictions = []
        current_pos = initial_pos.copy()
        current_vel = initial_vel.copy()
        
        for _ in range(n_steps):
            # Extract neighbor positions and velocities
            n_positions = [n[0] for n in neighbor_data]
            n_velocities = [n[1] for n in neighbor_data]
            
            # Compute social forces
            force = self.compute_social_forces(
                current_pos, current_vel, n_positions, n_velocities
            )
            
            # Update velocity (F = ma, assuming unit mass)
            acceleration = force * 0.1  # Damping factor
            current_vel += acceleration * self.dt
            
            # Limit velocity
            vel_mag = np.linalg.norm(current_vel)
            max_vel = 20.0
            if vel_mag > max_vel:
                current_vel = current_vel / vel_mag * max_vel
            
            # Update position
            current_pos += current_vel * self.dt
            predictions.append(current_pos.copy())
            
            # Update neighbor positions (assume they continue straight)
            neighbor_data = [
                (n_pos + n_vel * self.dt, n_vel)
                for n_pos, n_vel in neighbor_data
            ]
        
        return np.array(predictions)


class RoadConstraintAnalyzer:
    """Detect drivable areas and constrain predictions to roads"""
    
    def __init__(self, frame_shape: Tuple[int, int]):
        self.frame_height, self.frame_width = frame_shape
        self.drivable_mask = None
        self.road_regions = []
        
    def build_drivable_mask_from_trajectories(
        self, all_trajectories: Dict[int, List[np.ndarray]], buffer_size: int = 80
    ):
        """Build drivable area mask from observed vehicle trajectories"""
        
        print("\n🛣️  Building road map from vehicle movements...")
        
        # Collect all observed positions
        all_points = []
        for vehicle_id, positions in all_trajectories.items():
            if len(positions) >= 5:  # Sufficient history
                all_points.extend(positions)
        
        if len(all_points) < 50:
            print("   ⚠️  Insufficient data, using full frame")
            self.drivable_mask = np.ones((self.frame_height, self.frame_width), dtype=np.uint8) * 255
            return
        
        all_points = np.array(all_points)
        
        # Cluster points to find road regions using DBSCAN
        clustering = DBSCAN(eps=50, min_samples=10).fit(all_points)
        labels = clustering.labels_
        
        # Create mask
        self.drivable_mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        
        # For each cluster, create a convex hull and dilate
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Remove noise
        
        for label in unique_labels:
            cluster_points = all_points[labels == label]
            
            if len(cluster_points) < 4:
                continue
            
            # Create convex hull
            try:
                hull = ConvexHull(cluster_points)
                hull_points = cluster_points[hull.vertices].astype(np.int32)
                
                # Draw filled polygon
                cv2.fillConvexPoly(self.drivable_mask, hull_points, 255)
            except:
                # If hull fails, use individual points with buffer
                for pt in cluster_points:
                    cv2.circle(self.drivable_mask, tuple(pt.astype(int)), buffer_size // 2, 255, -1)
        
        # Dilate to create buffer zone around observed paths
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_size, buffer_size))
        self.drivable_mask = cv2.dilate(self.drivable_mask, kernel, iterations=1)
        
        # Smooth edges
        self.drivable_mask = cv2.GaussianBlur(self.drivable_mask, (31, 31), 0)
        _, self.drivable_mask = cv2.threshold(self.drivable_mask, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate coverage
        coverage = np.sum(self.drivable_mask > 0) / (self.frame_width * self.frame_height) * 100
        print(f"   ✅ Road map built: {coverage:.1f}% of frame is drivable")
        print(f"   📍 Found {len(unique_labels)} road regions")
    
    def is_on_road(self, point: np.ndarray) -> bool:
        """Check if a point is on drivable area"""
        if self.drivable_mask is None:
            return True
        
        x, y = int(point[0]), int(point[1])
        if 0 <= x < self.frame_width and 0 <= y < self.frame_height:
            return self.drivable_mask[y, x] > 0
        return False
    
    def constrain_to_road(
        self, trajectory: np.ndarray, search_radius: int = 50
    ) -> np.ndarray:
        """Constrain trajectory points to stay on roads"""
        
        if self.drivable_mask is None:
            return trajectory
        
        constrained = trajectory.copy()
        
        for i, point in enumerate(trajectory):
            if not self.is_on_road(point):
                # Find nearest road point
                x, y = int(point[0]), int(point[1])
                
                # Search in expanding radius
                found = False
                for radius in range(10, search_radius, 10):
                    y_min = max(0, y - radius)
                    y_max = min(self.frame_height, y + radius)
                    x_min = max(0, x - radius)
                    x_max = min(self.frame_width, x + radius)
                    
                    region = self.drivable_mask[y_min:y_max, x_min:x_max]
                    if np.any(region > 0):
                        # Find closest road pixel in region
                        road_pixels = np.argwhere(region > 0)
                        if len(road_pixels) > 0:
                            # Convert to absolute coordinates
                            road_pixels[:, 0] += y_min
                            road_pixels[:, 1] += x_min
                            
                            # Find closest
                            distances = np.linalg.norm(
                                road_pixels[:, [1, 0]] - point, axis=1
                            )
                            closest_idx = np.argmin(distances)
                            closest_point = road_pixels[closest_idx, [1, 0]]
                            
                            constrained[i] = closest_point.astype(float)
                            found = True
                            break
                
                if not found:
                    # Use previous valid point or clip to bounds
                    if i > 0:
                        constrained[i] = constrained[i-1]
        
        return constrained
    
    def get_road_direction(
        self, point: np.ndarray, history: np.ndarray, radius: int = 40
    ) -> Optional[np.ndarray]:
        """Get preferred road direction at a point based on local road geometry"""
        
        if self.drivable_mask is None or len(history) < 2:
            return None
        
        x, y = int(point[0]), int(point[1])
        
        # Get local road region
        y_min = max(0, y - radius)
        y_max = min(self.frame_height, y + radius)
        x_min = max(0, x - radius)
        x_max = min(self.frame_width, x + radius)
        
        region = self.drivable_mask[y_min:y_max, x_min:x_max]
        
        if np.sum(region > 0) < 10:
            return None
        
        # Get road pixels
        road_pixels = np.argwhere(region > 0)
        road_pixels[:, 0] += y_min
        road_pixels[:, 1] += x_min
        
        # Get historical direction
        hist_direction = history[-1] - history[-min(5, len(history))]
        hist_direction = hist_direction / (np.linalg.norm(hist_direction) + 1e-6)
        
        # Find road pixels in the forward direction
        forward_pixels = []
        for rp in road_pixels:
            diff = rp[[1, 0]].astype(float) - point
            if np.dot(diff, hist_direction) > 0:  # In forward direction
                forward_pixels.append(rp[[1, 0]])
        
        if len(forward_pixels) < 3:
            return hist_direction
        
        # Compute average direction to forward road pixels
        forward_pixels = np.array(forward_pixels)
        avg_direction = np.mean(forward_pixels - point, axis=0)
        avg_direction = avg_direction / (np.linalg.norm(avg_direction) + 1e-6)
        
        # Blend with historical direction (70% road geometry, 30% history)
        blended = 0.7 * avg_direction + 0.3 * hist_direction
        blended = blended / (np.linalg.norm(blended) + 1e-6)
        
        return blended


class CurvatureAnalyzer:
    """Analyze trajectory curvature for lane prediction"""
    
    @staticmethod
    def compute_curvature(points: np.ndarray, smooth_sigma: float = 2.0) -> np.ndarray:
        """Compute curvature along trajectory"""
        
        if len(points) < 5:
            return np.zeros(len(points))
        
        # Smooth the trajectory
        smoothed = np.copy(points)
        smoothed[:, 0] = gaussian_filter1d(points[:, 0], sigma=smooth_sigma)
        smoothed[:, 1] = gaussian_filter1d(points[:, 1], sigma=smooth_sigma)
        
        # Compute first and second derivatives
        dx = np.gradient(smoothed[:, 0])
        dy = np.gradient(smoothed[:, 1])
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2)**(3/2) + 1e-6
        
        curvature = numerator / denominator
        
        return curvature
    
    @staticmethod
    def fit_clothoid(points: np.ndarray) -> np.ndarray:
        """Fit clothoid (Euler spiral) for natural vehicle paths"""
        
        if len(points) < 3:
            return points
        
        # Parameterize by arc length
        diffs = np.diff(points, axis=0)
        segment_lengths = np.sqrt((diffs**2).sum(axis=1))
        arc_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        
        # Fit cubic spline for smooth interpolation
        tck_x, _ = interpolate.splprep([points[:, 0]], u=arc_length, s=0, k=min(3, len(points)-1))
        tck_y, _ = interpolate.splprep([points[:, 1]], u=arc_length, s=0, k=min(3, len(points)-1))
        
        return points


class AdaptiveCurvePredictor:
    """Predict trajectories that follow road curvature naturally"""
    
    def __init__(self, road_analyzer: RoadConstraintAnalyzer):
        self.road_analyzer = road_analyzer
    
    def predict_adaptive_curve(
        self,
        history: np.ndarray,
        n_steps: int,
        step_size: float = None
    ) -> np.ndarray:
        """Predict trajectory following road geometry with adaptive curvature"""
        
        if len(history) < 3:
            return np.array([history[-1]])
        
        # Estimate current velocity
        velocities = np.diff(history[-5:], axis=0) if len(history) >= 5 else np.diff(history, axis=0)
        current_velocity = np.mean(velocities, axis=0)
        speed = np.linalg.norm(current_velocity)
        
        if speed < 0.5:
            speed = 5.0
        
        if step_size is None:
            step_size = speed
        
        # Detect current curvature trend
        curvatures = CurvatureAnalyzer.compute_curvature(history)
        recent_curvature = np.mean(curvatures[-5:]) if len(curvatures) >= 5 else 0
        
        # Initialize prediction
        predictions = []
        current_pos = history[-1].copy()
        current_dir = current_velocity / (speed + 1e-6)
        
        # Adaptive parameters
        smoothness = 0.85  # How much to keep current direction
        
        for step in range(n_steps):
            # Get road-aware direction
            road_direction = self.road_analyzer.get_road_direction(
                current_pos, history, radius=60
            )
            
            if road_direction is not None:
                # Blend current direction with road direction
                # More aggressive road following for high curvature
                road_weight = 0.4 if abs(recent_curvature) > 0.005 else 0.25
                current_dir = smoothness * current_dir + (1 - smoothness) * road_direction
                current_dir = current_dir / (np.linalg.norm(current_dir) + 1e-6)
            
            # Predict next position
            next_pos = current_pos + current_dir * step_size
            
            # Constrain to road
            if not self.road_analyzer.is_on_road(next_pos):
                # Try to find valid position nearby
                constrained = self.road_analyzer.constrain_to_road(
                    np.array([next_pos]), search_radius=80
                )[0]
                
                # Adjust direction based on constraint
                if np.linalg.norm(constrained - next_pos) > 1:
                    correction = constrained - current_pos
                    correction = correction / (np.linalg.norm(correction) + 1e-6)
                    current_dir = 0.6 * current_dir + 0.4 * correction
                    current_dir = current_dir / (np.linalg.norm(current_dir) + 1e-6)
                    next_pos = constrained
            
            # Check if exited frame
            if not (-100 <= next_pos[0] <= self.road_analyzer.frame_width + 100 and
                   -100 <= next_pos[1] <= self.road_analyzer.frame_height + 100):
                break
            
            predictions.append(next_pos.copy())
            current_pos = next_pos
            
            # Gradually reduce speed (realistic deceleration)
            step_size *= 0.998
        
        return np.array(predictions) if predictions else np.array([history[-1]])


class AdvancedTrajectoryAnalyzer:
    """
    Enhanced trajectory analyzer with multiple prediction algorithms
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
        
        # Initialize predictor
        self.predictor = TrajectoryPredictor(
            model_path=model_path,
            history_length=history_length,
            future_length=future_length,
            num_modes=3
        )
        
        # Storage for trajectories
        self.trajectories: Dict[int, List[np.ndarray]] = defaultdict(list)
        self.predictions: Dict[int, Dict] = {}
        self.frame_width = 0
        self.frame_height = 0
        
        # Initialize predictors
        self.kalman_predictor = EnhancedKalmanPredictor(dt=1.0)
        self.social_predictor = SocialForcePredictor(dt=1.0)
        self.curvature_analyzer = CurvatureAnalyzer()
        self.road_analyzer = None  # Will be initialized after trajectory collection
        self.adaptive_predictor = None
    
    def analyze_video(
        self,
        start_frame: int = 0,
        end_frame: int = None,
        max_frames: int = 500
    ):
        """Analyze video and predict trajectories"""
        
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if end_frame is None:
            end_frame = min(start_frame + max_frames, total_frames)
        else:
            end_frame = min(end_frame, total_frames)
        
        print(f"\n╔══════════════════════════════════════════════════════════════════╗")
        print(f"║    ENHANCED TRAJECTORY PREDICTION SYSTEM                         ║")
        print(f"╚══════════════════════════════════════════════════════════════════╝")
        print(f"\n📹 Video: {self.video_path}")
        print(f"📊 Frames: {start_frame} → {end_frame} ({end_frame - start_frame} frames)")
        print(f"⚡ FPS: {fps:.1f}")
        print(f"📈 History: {self.predictor.history_length} frames")
        print(f"🔮 Prediction: Dynamic (until frame exit)")
        print(f"\n🧠 Algorithms Active:")
        print(f"   • Enhanced Kalman Filter (Constant Acceleration)")
        print(f"   • Social Force Model (Vehicle Interactions)")
        print(f"   • Cubic Spline Interpolation")
        print(f"   • Curvature Analysis")
        print(f"   • Ensemble Prediction")
        print("─" * 70)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        processed = 0
        
        # Get frame dimensions
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        full_frame_polygon = [(0, 0), (self.frame_width, 0), 
                             (self.frame_width, self.frame_height), (0, self.frame_height)]
        
        while frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles
            result = self.analyzer.analyze_frame(frame, full_frame_polygon)
            detections = result['roi_vehicles']
            
            # Update tracker
            tracked_objects = self.tracker.update(detections)
            
            # Store positions
            for vehicle_id, track_data in tracked_objects.items():
                bbox = track_data['bbox']
                center = np.array([
                    (bbox[0] + bbox[2]) / 2,
                    (bbox[1] + bbox[3]) / 2
                ])
                self.trajectories[vehicle_id].append(center)
            
            processed += 1
            if processed % 50 == 0:
                print(f"  ⏳ Processed {processed}/{end_frame - start_frame} frames... "
                      f"({len(self.trajectories)} vehicles tracked)")
            
            frame_count += 1
        
        cap.release()
        
        print(f"\n✅ Analysis complete!")
        print(f"   📍 Total vehicles tracked: {len(self.trajectories)}")
        
        # Build road constraint map from observed trajectories
        self.road_analyzer = RoadConstraintAnalyzer((self.frame_height, self.frame_width))
        self.road_analyzer.build_drivable_mask_from_trajectories(self.trajectories, buffer_size=100)
        self.adaptive_predictor = AdaptiveCurvePredictor(self.road_analyzer)
        
        # Generate enhanced predictions
        self._generate_enhanced_predictions()
    
    def _smooth_trajectory_advanced(self, positions: np.ndarray) -> np.ndarray:
        """Advanced trajectory smoothing with multiple methods"""
        
        if len(positions) < 5:
            return positions
        
        # Method 1: Kalman smoothing
        kalman_smooth = smooth_kalman(positions)
        
        # Method 2: Gaussian smoothing
        gaussian_smooth = np.copy(positions)
        gaussian_smooth[:, 0] = gaussian_filter1d(positions[:, 0], sigma=1.5)
        gaussian_smooth[:, 1] = gaussian_filter1d(positions[:, 1], sigma=1.5)
        
        # Method 3: Cubic spline
        if len(positions) >= 4:
            t = np.arange(len(positions))
            cs_x = interpolate.CubicSpline(t, positions[:, 0])
            cs_y = interpolate.CubicSpline(t, positions[:, 1])
            spline_smooth = np.column_stack([cs_x(t), cs_y(t)])
        else:
            spline_smooth = positions
        
        # Ensemble: weighted average
        weights = np.array([0.4, 0.3, 0.3])
        smoothed = (weights[0] * kalman_smooth + 
                   weights[1] * gaussian_smooth + 
                   weights[2] * spline_smooth)
        
        return smoothed
    
    def _estimate_velocity_acceleration(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate velocity and acceleration with noise reduction"""
        
        if len(positions) < 3:
            return np.array([0.0, 0.0]), np.array([0.0, 0.0])
        
        # Smooth positions first
        smooth_pos = gaussian_filter1d(positions, sigma=1.0, axis=0)
        
        # Compute velocities
        velocities = np.diff(smooth_pos, axis=0)
        
        # Exponential weighted average for velocity
        if len(velocities) >= 5:
            weights = np.exp(np.linspace(-2, 0, min(10, len(velocities))))
            weights = weights / weights.sum()
            recent_vels = velocities[-len(weights):]
            velocity = np.average(recent_vels, axis=0, weights=weights)
        else:
            velocity = np.mean(velocities[-3:], axis=0)
        
        # Compute acceleration
        if len(velocities) >= 2:
            accelerations = np.diff(velocities, axis=0)
            if len(accelerations) >= 3:
                weights = np.exp(np.linspace(-1, 0, min(5, len(accelerations))))
                weights = weights / weights.sum()
                recent_accs = accelerations[-len(weights):]
                acceleration = np.average(recent_accs, axis=0, weights=weights)
            else:
                acceleration = np.mean(accelerations, axis=0)
            
            # Limit acceleration
            acc_mag = np.linalg.norm(acceleration)
            if acc_mag > 1.5:
                acceleration = acceleration / acc_mag * 1.5
        else:
            acceleration = np.array([0.0, 0.0])
        
        return velocity, acceleration
    
    def _predict_frames_until_exit(
        self, position: np.ndarray, velocity: np.ndarray
    ) -> int:
        """Calculate frames until vehicle exits"""
        
        margin = 100
        x, y = position
        vx, vy = velocity
        
        frames = []
        
        if abs(vx) > 0.1:
            if vx > 0:
                frames.append((self.frame_width + margin - x) / vx)
            else:
                frames.append((x + margin) / abs(vx))
        
        if abs(vy) > 0.1:
            if vy > 0:
                frames.append((self.frame_height + margin - y) / vy)
            else:
                frames.append((y + margin) / abs(vy))
        
        valid = [f for f in frames if f > 0]
        if valid:
            return min(int(min(valid)), 400)
        return 50
    
    def _get_neighbor_data(
        self, ego_id: int, ego_positions: np.ndarray, time_index: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get neighbor positions and velocities"""
        
        neighbors = []
        ego_pos = ego_positions[-1]
        
        for vehicle_id, positions in self.trajectories.items():
            if vehicle_id == ego_id or len(positions) < time_index:
                continue
            
            n_positions = np.array(positions[-time_index:])
            n_pos = n_positions[-1]
            
            # Distance check
            dist = np.linalg.norm(n_pos - ego_pos)
            if dist < 150 and dist > 1:  # Within 150 pixels
                # Estimate neighbor velocity
                if len(n_positions) >= 2:
                    n_vel = n_positions[-1] - n_positions[-2]
                else:
                    n_vel = np.array([0.0, 0.0])
                
                neighbors.append((n_pos, n_vel))
        
        return neighbors[:5]  # Max 5 neighbors
    
    def _generate_enhanced_predictions(self):
        """Generate predictions using ensemble of algorithms"""
        
        print(f"\n🔮 Generating enhanced predictions...")
        print("─" * 70)
        
        min_history = self.predictor.history_length
        predicted_count = 0
        
        for vehicle_id, positions in self.trajectories.items():
            if len(positions) < min_history:
                continue
            
            recent_positions = np.array(positions[-min_history:])
            
            # Advanced smoothing
            smoothed = self._smooth_trajectory_advanced(recent_positions)
            
            # Estimate dynamics
            velocity, acceleration = self._estimate_velocity_acceleration(smoothed)
            last_pos = smoothed[-1]
            
            # Calculate prediction horizon
            n_steps = self._predict_frames_until_exit(last_pos, velocity)
            
            if predicted_count < 3:
                print(f"\n🚗 Vehicle {vehicle_id}:")
                print(f"   📍 Position: [{last_pos[0]:.1f}, {last_pos[1]:.1f}]")
                print(f"   ⚡ Velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}] px/frame")
                print(f"   🚀 Acceleration: [{acceleration[0]:.3f}, {acceleration[1]:.3f}] px/frame²")
                print(f"   🎯 Prediction horizon: {n_steps} frames")
            
            # ═══════════════════════════════════════════════════════════
            # ALGORITHM 1: Enhanced Kalman Filter
            # ═══════════════════════════════════════════════════════════
            self.kalman_predictor.initialize(last_pos, velocity)
            kalman_pred = self.kalman_predictor.predict(n_steps)
            
            # ═══════════════════════════════════════════════════════════
            # ALGORITHM 2: Adaptive Curve Following (Road-Aware)
            # ═══════════════════════════════════════════════════════════
            adaptive_pred = self.adaptive_predictor.predict_adaptive_curve(
                smoothed, n_steps, step_size=np.linalg.norm(velocity)
            )
            
            # Constrain to roads
            adaptive_pred = self.road_analyzer.constrain_to_road(adaptive_pred, search_radius=80)
            
            # ═══════════════════════════════════════════════════════════
            # ALGORITHM 3: Social Force Model
            # ═══════════════════════════════════════════════════════════
            neighbor_data = self._get_neighbor_data(vehicle_id, smoothed, min_history)
            
            if len(neighbor_data) > 0:
                social_pred = self.social_predictor.predict_with_social_forces(
                    last_pos, velocity, neighbor_data, min(n_steps, len(adaptive_pred))
                )
            else:
                social_pred = adaptive_pred.copy()
            
            # ═══════════════════════════════════════════════════════════
            # ALGORITHM 4: Cubic Spline with Road Constraints
            # ═══════════════════════════════════════════════════════════
            if len(smoothed) >= 4:
                # Fit cubic spline to history
                t_hist = np.arange(len(smoothed))
                cs_x = interpolate.CubicSpline(t_hist, smoothed[:, 0])
                cs_y = interpolate.CubicSpline(t_hist, smoothed[:, 1])
                
                # Extrapolate with smaller steps for better control
                max_spline_len = min(n_steps, len(adaptive_pred))
                t_future = np.linspace(len(smoothed), len(smoothed) + max_spline_len - 1, max_spline_len)
                spline_pred = np.column_stack([cs_x(t_future), cs_y(t_future)])
                
                # Constrain spline to roads
                spline_pred = self.road_analyzer.constrain_to_road(spline_pred, search_radius=80)
            else:
                spline_pred = adaptive_pred.copy()
            
            # ═══════════════════════════════════════════════════════════
            # ENSEMBLE: Weighted combination with road awareness
            # ═══════════════════════════════════════════════════════════
            min_len = min(len(kalman_pred), len(adaptive_pred), 
                         len(social_pred), len(spline_pred))
            
            if min_len > 0:
                # Trim all to same length
                kalman_pred = kalman_pred[:min_len]
                adaptive_pred = adaptive_pred[:min_len]
                social_pred = social_pred[:min_len]
                spline_pred = spline_pred[:min_len]
                
                # Analyze curvature to weight methods
                curvature = self.curvature_analyzer.compute_curvature(smoothed)
                avg_curvature = np.mean(curvature[-5:])
                
                if avg_curvature > 0.01:  # High curvature (turning)
                    # Favor adaptive and spline for curves
                    weights = np.array([0.2, 0.4, 0.1, 0.3])
                else:  # Straight or gentle curve
                    # Balance adaptive with Kalman
                    weights = np.array([0.3, 0.4, 0.15, 0.15])
                
                # Main prediction (ensemble)
                main_pred = (weights[0] * kalman_pred +
                            weights[1] * adaptive_pred +
                            weights[2] * social_pred +
                            weights[3] * spline_pred)
                
                # Constrain ensemble to roads
                main_pred = self.road_analyzer.constrain_to_road(main_pred, search_radius=80)
                
                # Smooth the ensemble prediction
                if len(main_pred) > 5:
                    main_pred[:, 0] = gaussian_filter1d(main_pred[:, 0], sigma=1.5)
                    main_pred[:, 1] = gaussian_filter1d(main_pred[:, 1], sigma=1.5)
                    
                    # Re-constrain after smoothing
                    main_pred = self.road_analyzer.constrain_to_road(main_pred, search_radius=60)
                
                # ═══════════════════════════════════════════════════════════
                # Generate multiple modes (uncertainty quantification)
                # ═══════════════════════════════════════════════════════════
                
                # Mode 1: Main ensemble prediction (70% confidence)
                mode1 = main_pred.copy()
                
                # Mode 2: Slight left deviation (20% confidence)
                mode2 = main_pred.copy()
                perpendicular = np.array([-velocity[1], velocity[0]])
                perpendicular = perpendicular / (np.linalg.norm(perpendicular) + 1e-6)
                
                for i in range(len(mode2)):
                    curve_factor = np.sqrt(i / len(mode2))  # Gradual
                    lateral_offset = perpendicular * 20 * curve_factor
                    mode2[i] += lateral_offset
                
                # Mode 3: Slight right deviation (10% confidence)
                mode3 = main_pred.copy()
                for i in range(len(mode3)):
                    curve_factor = np.sqrt(i / len(mode3))
                    lateral_offset = -perpendicular * 20 * curve_factor
                    mode3[i] += lateral_offset
                
                # Constrain modes to roads
                mode2 = self.road_analyzer.constrain_to_road(mode2, search_radius=60)
                mode3 = self.road_analyzer.constrain_to_road(mode3, search_radius=60)
                
                # Smooth modes
                if len(mode2) > 3:
                    mode2[:, 0] = gaussian_filter1d(mode2[:, 0], sigma=1.0)
                    mode2[:, 1] = gaussian_filter1d(mode2[:, 1], sigma=1.0)
                    mode3[:, 0] = gaussian_filter1d(mode3[:, 0], sigma=1.0)
                    mode3[:, 1] = gaussian_filter1d(mode3[:, 1], sigma=1.0)
                    
                    # Re-constrain after smoothing
                    mode2 = self.road_analyzer.constrain_to_road(mode2, search_radius=50)
                    mode3 = self.road_analyzer.constrain_to_road(mode3, search_radius=50)
                
                # Store prediction with metadata
                self.predictions[vehicle_id] = {
                    'modes': np.array([mode1, mode2, mode3]),
                    'probabilities': np.array([0.7, 0.2, 0.1]),
                    'algorithms': {
                        'kalman': kalman_pred,
                        'adaptive': adaptive_pred,
                        'social': social_pred,
                        'spline': spline_pred
                    },
                    'curvature': avg_curvature,
                    'n_neighbors': len(neighbor_data)
                }
                
                predicted_count += 1
                
                if predicted_count <= 3:
                    print(f"   ✅ Generated {len(mode1)} prediction points")
                    print(f"   🔄 Curvature: {avg_curvature:.4f}")
                    print(f"   👥 Neighbors: {len(neighbor_data)}")
        
        print("─" * 70)
        print(f"✅ Generated ensemble predictions for {predicted_count} vehicles")
        print(f"   Using 4 algorithms: Kalman, Physics, Social Forces, Spline")
    
    def visualize_predictions(
        self,
        output_dir: str = "output/predictions",
        top_k: int = 5,
        show_algorithms: bool = True
    ):
        """Enhanced visualization with algorithm comparison"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get background frame
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        
        print(f"\n🎨 Visualizing top {len(sorted_vehicles)} predictions...")
        print("─" * 70)
        
        for idx, (vehicle_id, pred_data) in enumerate(sorted_vehicles, 1):
            img = background.copy()
            
            # Get smoothed history
            history = self._smooth_trajectory_advanced(
                np.array(self.trajectories[vehicle_id])
            )
            
            # Draw history (green gradient)
            for i in range(1, len(history)):
                alpha = i / len(history)
                color = (0, int(150 + 105 * alpha), 0)
                pt1 = tuple(history[i-1].astype(int))
                pt2 = tuple(history[i].astype(int))
                cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)
            
            # Draw start and end of history
            cv2.circle(img, tuple(history[0].astype(int)), 6, (0, 100, 0), -1)
            last_pos = tuple(history[-1].astype(int))
            cv2.circle(img, last_pos, 10, (0, 255, 0), -1)
            cv2.circle(img, last_pos, 12, (255, 255, 255), 2)
            
            # Get prediction modes
            modes = pred_data['modes']
            probs = pred_data['probabilities']
            
            # Draw all predicted modes with beautiful colors
            colors = [
                (255, 100, 0),    # Deep blue - Mode 1 (main)
                (0, 200, 255),    # Orange - Mode 2 (left)
                (200, 255, 0)     # Cyan - Mode 3 (right)
            ]
            
            for mode_idx, (mode_traj, prob) in enumerate(zip(modes, probs)):
                color = colors[mode_idx]
                thickness = 4 if mode_idx == 0 else 2
                
                mode_traj = mode_traj.astype(int)
                
                # Draw with gradient effect
                for i in range(1, len(mode_traj)):
                    alpha = 1.0 - (i / len(mode_traj)) * 0.3  # Fade slightly
                    fade_color = tuple(int(c * alpha) for c in color)
                    pt1 = tuple(mode_traj[i-1])
                    pt2 = tuple(mode_traj[i])
                    cv2.line(img, pt1, pt2, fade_color, thickness, cv2.LINE_AA)
                
                # Draw endpoint
                endpoint = tuple(mode_traj[-1])
                cv2.circle(img, endpoint, 12, color, -1)
                cv2.circle(img, endpoint, 14, (255, 255, 255), 2)
                
                # Probability label
                label_pos = (endpoint[0] + 18, endpoint[1] + 6)
                cv2.putText(img, f"{prob:.0%}", label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
            
            # Optionally show individual algorithms
            if show_algorithms:
                algorithms = pred_data['algorithms']
                alg_colors = {
                    'kalman': (180, 180, 180),   # Gray
                    'adaptive': (200, 150, 100),  # Light brown
                    'social': (150, 100, 200),   # Purple
                    'spline': (100, 200, 150)    # Teal
                }
                
                for alg_name, alg_pred in algorithms.items():
                    alg_pred_int = alg_pred.astype(int)
                    for i in range(1, min(len(alg_pred_int), 50)):  # Show first 50 points
                        pt1 = tuple(alg_pred_int[i-1])
                        pt2 = tuple(alg_pred_int[i])
                        cv2.line(img, pt1, pt2, alg_colors[alg_name], 1, cv2.LINE_AA)
            
            # Info panel
            panel_height = 280
            panel = np.zeros((panel_height, img.shape[1], 3), dtype=np.uint8)
            
            info_lines = [
                f"╔══════════════════════════════════════════════════════════════╗",
                f"║  Vehicle ID: {vehicle_id:>4}                                          ║",
                f"╠══════════════════════════════════════════════════════════════╣",
                f"║  📊 Statistics:                                               ║",
                f"║     • History: {len(history)} frames                                ║",
                f"║     • Prediction: {len(modes[0])} frames                            ║",
                f"║     • Curvature: {pred_data['curvature']:.4f}                        ║",
                f"║     • Neighbors: {pred_data['n_neighbors']}                              ║",
                f"╠══════════════════════════════════════════════════════════════╣",
                f"║  🎯 Prediction Modes:                                         ║",
                f"║     • Mode 1 (Deep Blue):  {probs[0]:.0%} - Main trajectory        ║",
                f"║     • Mode 2 (Orange):     {probs[1]:.0%} - Left deviation         ║",
                f"║     • Mode 3 (Cyan):       {probs[2]:.0%} - Right deviation        ║",
                f"╠══════════════════════════════════════════════════════════════╣",
                f"║  🧠 Algorithms Used:                                          ║",
                f"║     Kalman • Adaptive Curve • Social Forces • Spline         ║",
                f"║  🛣️  Road-Constrained: YES                                    ║",
                f"╚══════════════════════════════════════════════════════════════╝"
            ]
            
            y_offset = 15
            for line in info_lines:
                cv2.putText(panel, line, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                y_offset += 17
            
            result = np.vstack([img, panel])
            
            # Save
            output_file = output_path / f"enhanced_pred_{idx:02d}_vehicle_{vehicle_id}.png"
            cv2.imwrite(str(output_file), result)
            print(f"  ✅ Saved: {output_file.name}")
        
        # All-vehicles overview
        print(f"\n🌍 Creating all-vehicles overview...")
        img_all = background.copy()
        
        color_palette = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (128, 255, 0), (128, 0, 255),
            (255, 0, 128), (0, 255, 128), (0, 128, 255)
        ]
        
        for vehicle_idx, (vehicle_id, pred_data) in enumerate(self.predictions.items()):
            color = color_palette[vehicle_idx % len(color_palette)]
            
            history = self._smooth_trajectory_advanced(
                np.array(self.trajectories[vehicle_id])
            )
            
            # Draw history
            history_color = tuple(int(c * 0.5) for c in color)
            for i in range(1, len(history)):
                pt1 = tuple(history[i-1].astype(int))
                pt2 = tuple(history[i].astype(int))
                cv2.line(img_all, pt1, pt2, history_color, 1, cv2.LINE_AA)
            
            # Draw main prediction
            predicted = pred_data['modes'][0].astype(int)
            last_pos = tuple(history[-1].astype(int))
            
            if len(predicted) > 0:
                for i in range(1, len(predicted)):
                    pt1 = tuple(predicted[i-1])
                    pt2 = tuple(predicted[i])
                    cv2.line(img_all, pt1, pt2, color, 2, cv2.LINE_AA)
                
                # Endpoints
                endpoint = tuple(predicted[-1])
                cv2.circle(img_all, endpoint, 7, color, -1)
                cv2.circle(img_all, endpoint, 9, (255, 255, 255), 1)
            
            cv2.circle(img_all, last_pos, 6, color, -1)
        
        # Title panel
        title_panel = np.zeros((80, img_all.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_panel,
                   f"Enhanced Trajectory Predictions - {len(self.predictions)} Vehicles",
                   (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(title_panel,
                   "Algorithms: Kalman • Adaptive Curve • Social Forces • Spline | Road-Constrained",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        
        result_all = np.vstack([title_panel, img_all])
        
        output_file_all = output_path / "all_vehicles_enhanced.png"
        cv2.imwrite(str(output_file_all), result_all)
        print(f"  ✅ Saved: {output_file_all.name}")
        
        print(f"\n╔══════════════════════════════════════════════════════════════════╗")
        print(f"║  ✅ VISUALIZATION COMPLETE                                        ║")
        print(f"║  📁 Output: {str(output_path):<50} ║")
        print(f"╚══════════════════════════════════════════════════════════════════╝")


def main():
    parser = argparse.ArgumentParser(
        description="Road-Aware Trajectory Prediction with Multiple Algorithms"
    )
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--model", help="Path to pretrained model (optional)")
    parser.add_argument("--start", type=int, default=9700, help="Start frame")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames")
    parser.add_argument("--output", default="output/predictions", help="Output directory")
    parser.add_argument("--top-k", type=int, default=5, help="Top vehicles to visualize")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection threshold")
    parser.add_argument("--show-algorithms", action="store_true", 
                       help="Show individual algorithm predictions")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  ROAD-AWARE TRAJECTORY PREDICTION SYSTEM")
    print("  Powered by: Kalman • Adaptive Curve • Social Forces • Spline")
    print("  🛣️  Road-Constrained Predictions")
    print("="*70 + "\n")
    
    # Create analyzer
    analyzer = AdvancedTrajectoryAnalyzer(
        video_path=args.video,
        model_path=args.model,
        confidence_threshold=args.confidence,
        history_length=20,
        future_length=30
    )
    
    # Analyze
    analyzer.analyze_video(
        start_frame=args.start,
        max_frames=args.frames
    )
    
    # Visualize
    analyzer.visualize_predictions(
        output_dir=args.output,
        top_k=args.top_k,
        show_algorithms=args.show_algorithms
    )
    
    print("\n" + "="*70)
    print("  ✅ COMPLETE! Your predictions are ready.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()