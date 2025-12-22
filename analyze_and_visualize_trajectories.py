#!/usr/bin/env python3
"""
Analyze video and visualize individual vehicle trajectories
Shows 5 vehicles with their trajectories in separate images
"""

#!/usr/bin/env python3
"""
Analyze video and visualize individual vehicle trajectories
Shows 5 vehicles with their trajectories in separate images
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys
from interactive_analytics import VehicleAnalyzer, VehicleTracker
from utils.trajectory import smooth_kalman, median_filter

def predict_future_trajectory(trajectory, num_future_points=15):
    """
    Predict future vehicle path using polynomial extrapolation
    
    Args:
        trajectory: List of [x, y] historical points
        num_future_points: Number of future points to predict
    
    Returns:
        List of predicted [x, y] points
    """
    if len(trajectory) < 5:
        return []
    
    # Use last 20 points for prediction (or all if less)
    recent = trajectory[-20:]
    
    # Extract x, y coordinates
    xs = np.array([p[0] for p in recent])
    ys = np.array([p[1] for p in recent])
    
    # Create time indices
    t = np.arange(len(recent))
    
    # Fit polynomial (degree 2 for curved paths)
    try:
        # Fit x(t) and y(t) separately
        px = np.polyfit(t, xs, deg=min(2, len(recent)-1))
        py = np.polyfit(t, ys, deg=min(2, len(recent)-1))
        
        # Predict future points
        future_t = np.arange(len(recent), len(recent) + num_future_points)
        future_x = np.polyval(px, future_t)
        future_y = np.polyval(py, future_t)
        
        # Combine into trajectory
        predicted = [[float(x), float(y)] for x, y in zip(future_x, future_y)]
        
        return predicted
    except:
        # Fallback: linear extrapolation
        if len(recent) >= 2:
            last = recent[-1]
            prev = recent[-2]
            dx = last[0] - prev[0]
            dy = last[1] - prev[1]
            
            predicted = []
            for i in range(1, num_future_points + 1):
                predicted.append([last[0] + dx * i, last[1] + dy * i])
            return predicted
        return []

def smooth_trajectory_kalman(trajectory, process_var=1.0, meas_var=4.0):
    """
    Smooth trajectory using Kalman filter to handle jitter
    
    Args:
        trajectory: List of [x, y] points
        process_var: Process noise (how much vehicle can move naturally)
        meas_var: Measurement noise (detection noise level)
    
    Returns:
        Smoothed trajectory using Kalman filter
    """
    if len(trajectory) < 2:
        return trajectory
    
    # Convert to tuples for Kalman filter
    points = [(float(p[0]), float(p[1])) for p in trajectory]
    
    # Apply Kalman filter
    smoothed = smooth_kalman(points, process_var=process_var, meas_var=meas_var)
    
    # Convert back to list format
    return [[x, y] for x, y in smoothed]

def analyze_video(video_path, num_frames=100):
    """
    Analyze video with reduced frame count
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to process (default: 100)
    """
    print("\n" + "="*70)
    print("VEHICLE TRAJECTORY ANALYSIS WITH KALMAN FILTER")
    print("="*70)
    print(f"Video: {video_path}")
    print(f"Processing: {num_frames} frames (faster processing)")
    print()
    
    # Initialize analyzer
    analyzer = VehicleAnalyzer(
        model_conf=0.20,  # Lower threshold for better detection
        use_sahi=True,
        sahi_slice_size=640
    )
    analyzer.load_model()
    
    # Initialize tracker with better parameters for jittery video
    tracker = VehicleTracker(min_iou=0.25, max_age=25)  # More forgiving tracking
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Use middle section of video for analysis
    start_frame = max(0, (total_frames // 2) - (num_frames // 2))
    end_frame = start_frame + num_frames
    
    print(f"Video info:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")
    print(f"  Analyzing frames: {start_frame} to {end_frame}")
    print()
    
    # Track vehicles
    vehicle_data = {}
    frame_data = {}  # Store which vehicles are in which frames
    
    print("Starting analysis...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        detections = analyzer._detect_vehicles(frame)
        
        # Update tracker
        tracks = tracker.update(detections)
        
        frame_data[frame_idx] = []
        
        # Store trajectory data with frame info
        for track_id, track_data in tracks.items():
            if track_id not in vehicle_data:
                vehicle_data[track_id] = {
                    'class': track_data['class_name'],
                    'trajectory': [],
                    'raw_trajectory': [],
                    'frames_tracked': 0,
                    'first_seen': frame_idx,
                    'last_seen': frame_idx,
                    'frame_indices': []
                }
            
            bbox = track_data['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            vehicle_data[track_id]['raw_trajectory'].append([center_x, center_y])
            vehicle_data[track_id]['frames_tracked'] += 1
            vehicle_data[track_id]['last_seen'] = frame_idx
            vehicle_data[track_id]['frame_indices'].append(frame_idx)
            
            frame_data[frame_idx].append(track_id)
        
        # Progress update
        if frame_idx % 10 == 0:
            progress = (frame_idx - start_frame) / num_frames * 100
            active_tracks = len([v for v in vehicle_data.values() if v['last_seen'] >= frame_idx - 5])
            print(f"  Progress: {progress:.1f}% - Frame {frame_idx}/{end_frame} - Active: {active_tracks} - Total tracked: {len(vehicle_data)}")
    
    cap.release()
    
    # Apply Kalman filter smoothing and calculate metrics
    print("\nApplying Kalman filter smoothing and predicting future paths...")
    for vid, vdata in vehicle_data.items():
        raw_traj = vdata['raw_trajectory']
        
        # Apply Kalman filter to handle jitter (more precise than moving average)
        if len(raw_traj) > 5:
            vdata['trajectory'] = smooth_trajectory_kalman(
                raw_traj, 
                process_var=2.0,  # Allow moderate natural movement
                meas_var=5.0      # Expect some detection noise
            )
        else:
            vdata['trajectory'] = raw_traj
        
        # Predict future trajectory
        vdata['predicted_trajectory'] = predict_future_trajectory(
            vdata['trajectory'], 
            num_future_points=15
        )
        
        # Calculate distance on smoothed trajectory
        if len(vdata['trajectory']) > 1:
            distance = 0
            for i in range(1, len(vdata['trajectory'])):
                dx = vdata['trajectory'][i][0] - vdata['trajectory'][i-1][0]
                dy = vdata['trajectory'][i][1] - vdata['trajectory'][i-1][1]
                distance += np.sqrt(dx**2 + dy**2)
            vdata['total_distance_pixels'] = distance
            
            # Calculate average velocity
            duration_frames = vdata['last_seen'] - vdata['first_seen']
            if duration_frames > 0:
                vdata['avg_velocity_px_per_frame'] = distance / duration_frames
            else:
                vdata['avg_velocity_px_per_frame'] = 0
        else:
            vdata['total_distance_pixels'] = 0
            vdata['avg_velocity_px_per_frame'] = 0
        
        # Calculate trajectory angle (direction)
        if len(vdata['trajectory']) > 1:
            start = vdata['trajectory'][0]
            end = vdata['trajectory'][-1]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            vdata['direction_angle'] = np.degrees(np.arctan2(dy, dx))
        else:
            vdata['direction_angle'] = 0
    
    print(f"\n✓ Analysis complete!")
    print(f"  Total vehicles tracked: {len(vehicle_data)}")
    
    results = {
        'vehicles': vehicle_data,
        'total_frames': num_frames,
        'frame_data': frame_data
    }
    
    return results

def select_best_vehicles(vehicles, num_vehicles=5):
    """
    Select vehicles with longest and most complete trajectories
    
    Args:
        vehicles: Dictionary of vehicle data
        num_vehicles: Number of vehicles to select
    
    Returns:
        List of (vehicle_id, vehicle_data) tuples
    """
    # Score vehicles based on trajectory quality
    scored_vehicles = []
    for vid, vdata in vehicles.items():
        trajectory = vdata.get('trajectory', [])
        
        # Minimum requirements
        if len(trajectory) < 15:  # At least 15 frames
            continue
        
        # Calculate quality score
        length_score = len(trajectory)
        distance_score = vdata.get('total_distance_pixels', 0) / 10  # Normalize
        
        # Penalize very short movements (likely stationary/parked vehicles)
        if vdata.get('total_distance_pixels', 0) < 50:
            continue
        
        total_score = length_score + distance_score
        scored_vehicles.append((vid, vdata, total_score))
    
    # Sort by score
    scored_vehicles.sort(key=lambda x: x[2], reverse=True)
    
    # Return top N
    return [(vid, vdata) for vid, vdata, score in scored_vehicles[:num_vehicles]]

def visualize_trajectories(video_path, vehicles, output_dir):
    """
    Create separate trajectory images for each vehicle with prediction
    
    Args:
        video_path: Path to video file
        vehicles: List of (vehicle_id, vehicle_data) tuples
        output_dir: Directory to save images
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get video frame from the middle of tracking (not just first frame)
    cap = cv2.VideoCapture(video_path)
    
    print("\n" + "="*70)
    print("CREATING TRAJECTORY VISUALIZATIONS WITH PREDICTION")
    print("="*70)
    
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Cyan
        (255, 0, 255),    # Magenta
    ]
    
    for idx, (vehicle_id, vehicle_data) in enumerate(vehicles):
        trajectory = vehicle_data.get('trajectory', [])
        predicted = vehicle_data.get('predicted_trajectory', [])
        
        if len(trajectory) < 2:
            print(f"Skipping {vehicle_id} - insufficient trajectory points")
            continue
        
        # Get frame from middle of tracking for better context
        middle_frame = (vehicle_data['first_seen'] + vehicle_data['last_seen']) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
        ret, background_frame = cap.read()
        
        if not ret:
            # Fallback to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, vehicle_data['first_seen'])
            ret, background_frame = cap.read()
        
        if not ret:
            background_frame = np.ones((2160, 3840, 3), dtype=np.uint8) * 50
        
        first_frame = vehicle_data['first_seen']
        last_frame = vehicle_data['last_seen']
        duration_sec = (last_frame - first_frame) / 25.0
        
        print(f"\n{'='*60}")
        print(f"Vehicle {idx+1}: ID {vehicle_id} (Kalman + Prediction)")
        print(f"{'='*60}")
        print(f"  Class: {vehicle_data['class']}")
        print(f"  Frames tracked: {vehicle_data['frames_tracked']}")
        print(f"  Active frames: {first_frame} → {last_frame} ({duration_sec:.2f}s)")
        print(f"  Historical points: {len(trajectory)} (smoothed)")
        print(f"  Predicted points: {len(predicted)}")
        print(f"  Distance traveled: {vehicle_data['total_distance_pixels']:.1f} pixels")
        print(f"  Avg velocity: {vehicle_data.get('avg_velocity_px_per_frame', 0):.2f} px/frame")
        print(f"  Direction: {vehicle_data.get('direction_angle', 0):.1f}°")
        print(f"  Smoothing: Kalman Filter + Polynomial Prediction")
        
        # Create visualization
        viz = background_frame.copy()
        
        # Draw trajectory path with gradient color (HISTORICAL - solid)
        color = colors[idx % len(colors)]
        points = np.array(trajectory, dtype=np.int32)
        
        # Draw thicker line for main trajectory
        cv2.polylines(viz, [points], False, color, 4)
        
        # Draw PREDICTED trajectory (dashed line in different color)
        if len(predicted) > 0:
            # Predicted color is lighter/translucent version
            pred_color = tuple(int(c * 0.6) for c in color)
            pred_points = np.array(predicted, dtype=np.int32)
            
            # Draw dashed line for prediction
            for i in range(len(pred_points) - 1):
                if i % 2 == 0:  # Dashed effect
                    cv2.line(viz, tuple(pred_points[i]), tuple(pred_points[i+1]), pred_color, 3)
            
            # Draw prediction endpoint (yellow circle)
            if len(pred_points) > 0:
                pred_end_x, pred_end_y = pred_points[-1]
                cv2.circle(viz, (int(pred_end_x), int(pred_end_y)), 20, (0, 255, 255), -1)
                cv2.circle(viz, (int(pred_end_x), int(pred_end_y)), 20, (255, 255, 255), 3)
                cv2.putText(viz, 'PREDICTED', (int(pred_end_x)-55, int(pred_end_y)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
                cv2.putText(viz, 'PREDICTED', (int(pred_end_x)-55, int(pred_end_y)-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw direction indicators along trajectory (arrows)
        for i in range(0, len(trajectory) - 1, max(1, len(trajectory) // 10)):
            if i + 1 < len(trajectory):
                p1 = (int(trajectory[i][0]), int(trajectory[i][1]))
                p2 = (int(trajectory[i+1][0]), int(trajectory[i+1][1]))
                cv2.arrowedLine(viz, p1, p2, color, 2, tipLength=0.3)
        
        # Draw points along historical trajectory with increasing size
        for i, (x, y) in enumerate(trajectory):
            progress = i / len(trajectory)
            
            # Size increases along trajectory
            radius = int(3 + progress * 5)
            
            # Draw point with white border
            cv2.circle(viz, (int(x), int(y)), radius + 1, (255, 255, 255), -1)
            cv2.circle(viz, (int(x), int(y)), radius, color, -1)
        
        # Draw start point (large green circle with label)
        start_x, start_y = trajectory[0]
        cv2.circle(viz, (int(start_x), int(start_y)), 20, (0, 255, 0), -1)
        cv2.circle(viz, (int(start_x), int(start_y)), 20, (255, 255, 255), 3)
        
        # Frame number at start
        start_frame_text = f"F:{vehicle_data['first_seen']}"
        cv2.putText(viz, 'START', (int(start_x)-35, int(start_y)-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
        cv2.putText(viz, 'START', (int(start_x)-35, int(start_y)-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(viz, start_frame_text, (int(start_x)-30, int(start_y)+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5)
        cv2.putText(viz, start_frame_text, (int(start_x)-30, int(start_y)+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw end point (large red circle with label)
        end_x, end_y = trajectory[-1]
        cv2.circle(viz, (int(end_x), int(end_y)), 20, (0, 0, 255), -1)
        cv2.circle(viz, (int(end_x), int(end_y)), 20, (255, 255, 255), 3)
        
        # Frame number at end
        end_frame_text = f"F:{vehicle_data['last_seen']}"
        cv2.putText(viz, 'END', (int(end_x)-25, int(end_y)-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 6)
        cv2.putText(viz, 'END', (int(end_x)-25, int(end_y)-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(viz, end_frame_text, (int(end_x)-30, int(end_y)+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 5)
        cv2.putText(viz, end_frame_text, (int(end_x)-30, int(end_y)+35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add comprehensive info panel with dark background
        first_frame = vehicle_data['first_seen']
        last_frame = vehicle_data['last_seen']
        duration_sec = (last_frame - first_frame) / 25.0
        predicted_frames = len(predicted)
        predicted_duration = predicted_frames / 25.0
        
        # Create dark semi-transparent overlay for text
        overlay = viz.copy()
        cv2.rectangle(overlay, (5, 5), (600, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, viz, 0.3, 0, viz)
        
        # Add border
        cv2.rectangle(viz, (5, 5), (600, 280), color, 3)
        
        info_text = [
            f"VEHICLE ID: {vehicle_id}",
            f"Class: {vehicle_data['class'].upper()}",
            f"",
            f"Historical Tracking:",
            f"  First seen: Frame {first_frame}",
            f"  Last seen:  Frame {last_frame}",
            f"  Duration:   {duration_sec:.2f}s ({vehicle_data['frames_tracked']} frames)",
            f"",
            f"Movement:",
            f"  Distance:   {vehicle_data['total_distance_pixels']:.1f} pixels",
            f"  Velocity:   {vehicle_data.get('avg_velocity_px_per_frame', 0):.2f} px/frame",
            f"  Direction:  {vehicle_data.get('direction_angle', 0):.1f} degrees",
            f"",
            f"Future Prediction:",
            f"  Next frames: {predicted_frames} (~{predicted_duration:.2f}s)",
            f"  Algorithm:   Polynomial Extrapolation",
            f"",
            f"Smoothing: KALMAN FILTER",
        ]
        
        y_offset = 30
        for i, line in enumerate(info_text):
            if line == "":
                y_offset += 10
                continue
            
            # Determine font size and weight
            if i == 0:  # Title
                font_scale = 0.9
                thickness = 2
                text_color = color
            elif ":" not in line and line.strip():  # Section headers
                font_scale = 0.7
                thickness = 2
                text_color = (200, 200, 200)
            else:  # Regular text
                font_scale = 0.6
                thickness = 1
                text_color = (255, 255, 255)
            
            cv2.putText(viz, line, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
            y_offset += 30 if i == 0 else 25
        
        # Save image
        output_path = output_dir / f"trajectory_{idx+1}_{vehicle_id}_predicted.png"
        cv2.imwrite(str(output_path), viz)
        print(f"  ✓ Saved: {output_path}")
    
    cap.release()
    
    print("\n" + "="*70)
    print(f"✓ ALL TRAJECTORIES WITH PREDICTIONS SAVED TO: {output_dir}")
    print("="*70)

def main():
    """Main function"""
    # Hardcoded video path
    video_path = r"C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"✗ Video not found: {video_path}")
        print("\nUsage:")
        print(f"  python {sys.argv[0]} [video_path]")
    # Analyze video (100 frames for faster processing with Kalman filter)
    results = analyze_video(str(video_path), num_frames=100)
    # Analyze video (150 frames for better trajectories)
    results = analyze_video(str(video_path), num_frames=150)
    
    # Select best 5 vehicles
    best_vehicles = select_best_vehicles(results['vehicles'], num_vehicles=5)
    
    if len(best_vehicles) == 0:
        print("\n✗ No vehicles with sufficient trajectories found")
        return
    
    print(f"\nSelected {len(best_vehicles)} vehicles for visualization:")
    for idx, (vid, vdata) in enumerate(best_vehicles):
        duration = (vdata['last_seen'] - vdata['first_seen']) / 25.0
        print(f"  {idx+1}. Vehicle {vid}")
        print(f"      Class: {vdata['class']}")
        print(f"      Tracked: {vdata['frames_tracked']} frames ({duration:.2f}s)")
        print(f"      Distance: {vdata['total_distance_pixels']:.1f} pixels")
        print(f"      Frames: {vdata['first_seen']} → {vdata['last_seen']}")
        print()
    
    # Create trajectory visualizations
    output_dir = Path("output/trajectories")
    visualize_trajectories(str(video_path), best_vehicles, output_dir)
    
    # Save analysis data
    analysis_output = {
        'video_path': str(video_path),
        'vehicles_analyzed': len(results['vehicles']),
        'vehicles_visualized': len(best_vehicles),
        'selected_vehicles': {
            vid: {
                'class': vdata['class'],
                'frames_tracked': vdata['frames_tracked'],
                'first_frame': vdata['first_seen'],
                'last_frame': vdata['last_seen'],
                'duration_seconds': (vdata['last_seen'] - vdata['first_seen']) / 25.0,
                'distance_pixels': vdata['total_distance_pixels'],
                'velocity_px_per_frame': vdata.get('avg_velocity_px_per_frame', 0),
                'direction_degrees': vdata.get('direction_angle', 0),
                'trajectory_points': len(vdata.get('trajectory', []))
            }
            for vid, vdata in best_vehicles
        }
    }
    
    analysis_path = output_dir / 'analysis_summary.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_output, f, indent=2)
    
    print(f"\n✓ Analysis summary saved: {analysis_path}")
    print(f"\n{'='*70}")
    print("✓ TRAJECTORY VISUALIZATION COMPLETE!")
    print(f"{'='*70}")
    print("\nGenerated files:")
    print(f"  📁 {output_dir.absolute()}/")
    for i in range(len(best_vehicles)):
        print(f"    📊 trajectory_{i+1}_*.png")
    print(f"    📄 analysis_summary.json")
    print(f"\nYou can now view these images to see precise vehicle trajectories!")
    print(f"Each image shows:")
    print(f"  • Kalman-filtered smooth path (handles jitter optimally)")
    print(f"  • SOLID LINE = Historical path (where vehicle was)")
    print(f"  • DASHED LINE = Predicted path (where vehicle will go)")
    print(f"  • GREEN circle = Start position with frame number")
    print(f"  • RED circle = Last known position with frame number")
    print(f"  • YELLOW circle = Predicted future endpoint")
    print(f"  • Direction arrows along the path")
    print(f"\n🎯 Algorithms Applied:")
    print(f"   - Kalman Filter: Statistical optimal smoothing")
    print(f"   - Polynomial Extrapolation: Future path prediction")
    print(f"   - Process variance: 2.0 | Measurement variance: 5.0")

if __name__ == "__main__":
    main()
