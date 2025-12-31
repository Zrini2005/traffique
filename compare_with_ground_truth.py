#!/usr/bin/env python3
"""
Compare generated trajectories with ground truth CSV
- Generate trajectories with light smoothing
- Match vehicles spatially and temporally
- Plot X vs Time and Y vs Time comparisons
- Calculate comprehensive metrics (MAE, RMSE, etc.)
"""
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from trajectory_tracker import VehicleTrajectoryTracker

def smooth_trajectory_light(points, sigma=1.0):
    """Apply very light Gaussian smoothing to reduce jitter"""
    if len(points) < 3:
        return points
    
    points = np.array(points)
    smoothed = points.copy()
    
    # Very light smoothing - sigma=1.0 is minimal
    smoothed[:, 0] = gaussian_filter1d(points[:, 0], sigma=sigma)
    smoothed[:, 1] = gaussian_filter1d(points[:, 1], sigma=sigma)
    
    return smoothed

def generate_trajectory_csv(video_path, num_frames=1000, roi_polygon=None, output_csv="output/generated_trajectories.csv", use_sahi=True):
    """Generate trajectory CSV with light smoothing"""
    
    print("="*70)
    print("GENERATING TRAJECTORIES")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print(f"Frames: {num_frames}")
    print(f"SAHI: {'Enabled' if use_sahi else 'Disabled'}")
    print(f"Output: {output_csv}\n")
    
    tracker = VehicleTrajectoryTracker(
        video_path=video_path,
        confidence_threshold=0.3,
        min_trajectory_length=10,
        roi_polygon=roi_polygon,
        use_sahi=use_sahi
    )
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    vehicle_trajectories = defaultdict(list)  # vehicle_id -> list of (frame, x, y)
    frame_count = 0
    
    print("Tracking vehicles...")
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = tracker.analyzer.analyze_frame(frame, roi_polygon)
        detections = result['roi_vehicles']
        tracked_objects = tracker.tracker.update(detections)
        
        for vehicle_id, track_data in tracked_objects.items():
            if track_data['age'] == 0:  # Active detection
                bbox = track_data['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                vehicle_trajectories[vehicle_id].append((frame_count, center_x, center_y))
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{num_frames} frames...")
    
    cap.release()
    
    # Apply light smoothing and create CSV
    print(f"\nApplying light smoothing and creating CSV...")
    
    rows = []
    for vehicle_id, trajectory in vehicle_trajectories.items():
        if len(trajectory) < 10:  # Skip very short tracks
            continue
        
        frames, xs, ys = zip(*trajectory)
        
        # Light smoothing
        points = np.column_stack([xs, ys])
        smoothed = smooth_trajectory_light(points, sigma=1.5)
        
        # Create CSV rows
        for i, frame_num in enumerate(frames):
            time = frame_num / fps
            rows.append({
                'Frame': frame_num,
                'Time': time,
                'VehicleID': vehicle_id,
                'X_pixel': smoothed[i, 0],
                'Y_pixel': smoothed[i, 1]
            })
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['VehicleID', 'Frame'])
    
    Path(output_csv).parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(output_csv, index=False)
    
    print(f"✅ Generated CSV: {output_csv}")
    print(f"   Vehicles: {df['VehicleID'].nunique()}")
    print(f"   Total points: {len(df)}")
    
    return df

def match_vehicles(df_generated, df_gt, min_overlap_frames=30, max_spatial_distance=80):
    """Match vehicles between generated and ground truth based on spatial-temporal overlap"""
    
    print("\n" + "="*70)
    print("MATCHING VEHICLES")
    print("="*70)
    
    gen_vehicles = df_generated['VehicleID'].unique()
    gt_vehicles = df_gt['VehicleID'].unique()
    
    print(f"\nGenerated vehicles: {len(gen_vehicles)}")
    print(f"Ground truth vehicles: {len(gt_vehicles)}")
    
    matches = []
    
    for gen_id in gen_vehicles:
        gen_data = df_generated[df_generated['VehicleID'] == gen_id]
        gen_frames = set(gen_data['Frame'])
        gen_center = (gen_data['X_pixel'].mean(), gen_data['Y_pixel'].mean())
        
        best_match = None
        best_score = 0
        
        for gt_id in gt_vehicles:
            gt_data = df_gt[df_gt['VehicleID'] == gt_id]
            gt_frames = set(gt_data['Frame'])
            gt_center = (gt_data['X_pixel'].mean(), gt_data['Y_pixel'].mean())
            
            # Calculate temporal overlap
            overlap_frames = gen_frames.intersection(gt_frames)
            if len(overlap_frames) < min_overlap_frames:
                continue
            
            # Calculate spatial distance
            spatial_dist = np.sqrt((gen_center[0] - gt_center[0])**2 + 
                                  (gen_center[1] - gt_center[1])**2)
            
            if spatial_dist > max_spatial_distance:
                continue
            
            # Score based on overlap and proximity
            score = len(overlap_frames) / spatial_dist
            
            if score > best_score:
                best_score = score
                best_match = (gen_id, gt_id, len(overlap_frames), spatial_dist)
        
        if best_match:
            matches.append(best_match)
    
    print(f"\n✅ Found {len(matches)} matches")
    print(f"   Match rate: {len(matches)/len(gt_vehicles)*100:.1f}% of ground truth vehicles\n")
    
    if matches:
        print("Top 5 matches:")
        for i, (gen_id, gt_id, overlap, dist) in enumerate(sorted(matches, key=lambda x: x[2], reverse=True)[:5], 1):
            print(f"  {i}. Gen:{gen_id} ↔ GT:{gt_id} | Overlap:{overlap} frames | Distance:{dist:.1f}px")
    
    return matches

def calculate_metrics(df_generated, df_gt, gen_id, gt_id):
    """Calculate comprehensive error metrics for a matched pair"""
    
    gen_data = df_generated[df_generated['VehicleID'] == gen_id].sort_values('Frame')
    gt_data = df_gt[df_gt['VehicleID'] == gt_id].sort_values('Frame')
    
    # Get common frames
    common_frames = sorted(set(gen_data['Frame']).intersection(set(gt_data['Frame'])))
    
    gen_common = gen_data[gen_data['Frame'].isin(common_frames)].sort_values('Frame')
    gt_common = gt_data[gt_data['Frame'].isin(common_frames)].sort_values('Frame')
    
    # Calculate errors
    x_errors = np.abs(gen_common['X_pixel'].values - gt_common['X_pixel'].values)
    y_errors = np.abs(gen_common['Y_pixel'].values - gt_common['Y_pixel'].values)
    euclidean_errors = np.sqrt((gen_common['X_pixel'].values - gt_common['X_pixel'].values)**2 +
                                (gen_common['Y_pixel'].values - gt_common['Y_pixel'].values)**2)
    
    metrics = {
        'gen_id': gen_id,
        'gt_id': gt_id,
        'frames': len(common_frames),
        'x_mae': x_errors.mean(),
        'x_rmse': np.sqrt((x_errors**2).mean()),
        'x_max': x_errors.max(),
        'y_mae': y_errors.mean(),
        'y_rmse': np.sqrt((y_errors**2).mean()),
        'y_max': y_errors.max(),
        'euclidean_mae': euclidean_errors.mean(),
        'euclidean_rmse': np.sqrt((euclidean_errors**2).mean()),
        'euclidean_max': euclidean_errors.max()
    }
    
    return metrics, gen_common, gt_common

def plot_comparison(gen_data, gt_data, gen_id, gt_id, metrics, output_path):
    """Create X vs Time and Y vs Time comparison plots"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    fig.suptitle(f'Trajectory Comparison: Gen {gen_id} vs GT {gt_id}\n'
                 f'Euclidean MAE: {metrics["euclidean_mae"]:.2f}px | '
                 f'RMSE: {metrics["euclidean_rmse"]:.2f}px | '
                 f'Frames: {metrics["frames"]}',
                 fontsize=14, fontweight='bold')
    
    # X vs Time
    ax1.plot(gen_data['Time'], gen_data['X_pixel'], 'b-o', label='Generated', 
             markersize=3, linewidth=2, alpha=0.7)
    ax1.plot(gt_data['Time'], gt_data['X_pixel'], 'r-s', label='Ground Truth', 
             markersize=3, linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (seconds)', fontsize=11)
    ax1.set_ylabel('X Position (pixels)', fontsize=11)
    ax1.set_title(f'X Position vs Time (MAE: {metrics["x_mae"]:.2f}px, RMSE: {metrics["x_rmse"]:.2f}px)', 
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Y vs Time
    ax2.plot(gen_data['Time'], gen_data['Y_pixel'], 'b-o', label='Generated', 
             markersize=3, linewidth=2, alpha=0.7)
    ax2.plot(gt_data['Time'], gt_data['Y_pixel'], 'r-s', label='Ground Truth', 
             markersize=3, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Y Position (pixels)', fontsize=11)
    ax2.set_title(f'Y Position vs Time (MAE: {metrics["y_mae"]:.2f}px, RMSE: {metrics["y_rmse"]:.2f}px)', 
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Invert Y axis for image coordinates
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate_video_overlay(video_path, num_frames, roi_polygon, output_path, use_sahi=True):
    """Generate video with vehicle path overlays"""
    
    print("\n" + "="*70)
    print("GENERATING VIDEO OVERLAY")
    print("="*70)
    print(f"\nCreating video with path visualization...")
    print(f"Output: {output_path}\n")
    
    # Setup tracker
    tracker = VehicleTrajectoryTracker(
        video_path=video_path,
        confidence_threshold=0.3,
        min_trajectory_length=5,
        roi_polygon=roi_polygon,
        use_sahi=use_sahi
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Colors for different vehicles
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
    ]
    
    # Store trajectory history
    vehicle_paths = defaultdict(list)
    vehicle_colors = {}
    vehicle_classes = {}
    color_idx = 0
    
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        result = tracker.analyzer.analyze_frame(frame, roi_polygon)
        detections = result['roi_vehicles']
        
        # Update tracker
        tracked_objects = tracker.tracker.update(detections)
        
        # Update paths for each tracked vehicle
        for vehicle_id, track_data in tracked_objects.items():
            if track_data['age'] == 0:
                bbox = track_data['bbox']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                vehicle_paths[vehicle_id].append((center_x, center_y))
                
                if vehicle_id not in vehicle_colors:
                    vehicle_colors[vehicle_id] = colors[color_idx % len(colors)]
                    color_idx += 1
                
                if vehicle_id not in vehicle_classes:
                    vehicle_classes[vehicle_id] = track_data.get('class_name', 'vehicle')
        
        # Draw all paths on frame
        for vehicle_id, path in vehicle_paths.items():
            if len(path) > 1:
                color = vehicle_colors[vehicle_id]
                
                # Apply adaptive smoothing for visualization only
                class_name = vehicle_classes.get(vehicle_id, 'vehicle')
                if class_name in ['truck', 'bus'] and len(path) >= 5:
                    # 3-point running average for large vehicles
                    smoothed_path = []
                    for i in range(len(path)):
                        if i == 0:
                            smoothed_path.append(path[i])
                        elif i == len(path) - 1:
                            smoothed_path.append(path[i])
                        else:
                            avg_x = (path[i-1][0] + path[i][0] + path[i+1][0]) // 3
                            avg_y = (path[i-1][1] + path[i][1] + path[i+1][1]) // 3
                            smoothed_path.append((avg_x, avg_y))
                    draw_path = smoothed_path
                else:
                    draw_path = path
                
                # Draw path lines
                for i in range(len(draw_path) - 1):
                    cv2.line(frame, draw_path[i], draw_path[i + 1], color, 2)
                
                # Draw current position
                if draw_path:
                    current_pos = draw_path[-1]
                    cv2.circle(frame, current_pos, 5, color, -1)
                    cv2.putText(frame, f"ID:{vehicle_id}", 
                              (current_pos[0] + 10, current_pos[1] - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"  Video: {frame_count}/{num_frames} frames processed...")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Video saved: {output_path}")
    print(f"   Total vehicles tracked: {len(vehicle_paths)}")

def main():
    import sys
    import argparse
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare generated trajectories with ground truth')
    parser.add_argument('--video', type=str, default="/mnt/c/Users/srini/Downloads/D2F1_stab.mp4",
                        help='Path to video file')
    parser.add_argument('--frames', type=int, default=1000,
                        help='Number of frames to process (default: 1000)')
    parser.add_argument('--use-sahi', action='store_true', default=True,
                        help='Use SAHI for detection (default: True)')
    parser.add_argument('--no-sahi', dest='use_sahi', action='store_false',
                        help='Disable SAHI for detection')
    parser.add_argument('--gt', type=str, default="/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv",
                        help='Path to ground truth CSV')
    parser.add_argument('--generate-video', action='store_true',
                        help='Generate video overlay with vehicle paths')
    
    args = parser.parse_args()
    
    # Configuration
    video_path = args.video
    gt_csv_path = args.gt
    num_frames = args.frames
    use_sahi = args.use_sahi
    generate_video = args.generate_video
    
    # Auto-generate output directory with counter
    counter = 1
    while True:
        output_dir = Path(f"output/comparison_{counter:03d}")
        if not output_dir.exists():
            break
        counter += 1
    
    output_dir.mkdir(exist_ok=True, parents=True)
    generated_csv = output_dir / "generated_trajectories.csv"
    
    roi_polygon = [(4, 873), (3827, 877), (3831, 1086), (4, 1071)]  # Road ROI
    
    print("\n" + "="*70)
    print("TRAJECTORY COMPARISON WITH GROUND TRUTH")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print(f"Ground Truth: {gt_csv_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Frames: {num_frames}")
    print(f"SAHI: {'Enabled' if use_sahi else 'Disabled'}\n")
    
    # Step 1: Generate trajectories
    df_generated = generate_trajectory_csv(video_path, num_frames, roi_polygon, generated_csv, use_sahi)
    
    # Step 2: Load ground truth
    print("\nLoading ground truth...")
    df_gt = pd.read_csv(gt_csv_path)
    
    # Filter GT to same frame range
    min_frame = df_generated['Frame'].min()
    max_frame = df_generated['Frame'].max()
    df_gt = df_gt[(df_gt['Frame'] >= min_frame) & (df_gt['Frame'] <= max_frame)]
    
    print(f"Ground truth: {df_gt['VehicleID'].nunique()} vehicles, {len(df_gt)} points")
    
    # Step 3: Match vehicles
    matches = match_vehicles(df_generated, df_gt)
    
    if not matches:
        print("\n❌ No matches found! Check frame alignment or ROI settings.")
        return
    
    # Step 4: Calculate metrics and generate plots
    print("\n" + "="*70)
    print("CALCULATING METRICS & GENERATING PLOTS")
    print("="*70)
    
    all_metrics = []
    
    for i, (gen_id, gt_id, _, _) in enumerate(matches, 1):
        metrics, gen_common, gt_common = calculate_metrics(df_generated, df_gt, gen_id, gt_id)
        all_metrics.append(metrics)
        
        # Generate plot for each match
        plot_path = output_dir / f"comparison_{i}_gen{gen_id}_gt{gt_id}.png"
        plot_comparison(gen_common, gt_common, gen_id, gt_id, metrics, plot_path)
        
        print(f"\n[{i}/{len(matches)}] Gen:{gen_id} ↔ GT:{gt_id}")
        print(f"  Euclidean: MAE={metrics['euclidean_mae']:.2f}px, RMSE={metrics['euclidean_rmse']:.2f}px")
        print(f"  X-axis: MAE={metrics['x_mae']:.2f}px, RMSE={metrics['x_rmse']:.2f}px")
        print(f"  Y-axis: MAE={metrics['y_mae']:.2f}px, RMSE={metrics['y_rmse']:.2f}px")
    
    # Step 5: Save metrics summary
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    
    # Step 6: Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    print(f"\n📊 Euclidean Distance Error:")
    print(f"   MAE:  {metrics_df['euclidean_mae'].mean():.2f} ± {metrics_df['euclidean_mae'].std():.2f} px")
    print(f"   RMSE: {metrics_df['euclidean_rmse'].mean():.2f} ± {metrics_df['euclidean_rmse'].std():.2f} px")
    print(f"   Max:  {metrics_df['euclidean_max'].max():.2f} px")
    
    print(f"\n📏 X-Axis Error:")
    print(f"   MAE:  {metrics_df['x_mae'].mean():.2f} ± {metrics_df['x_mae'].std():.2f} px")
    print(f"   RMSE: {metrics_df['x_rmse'].mean():.2f} ± {metrics_df['x_rmse'].std():.2f} px")
    
    print(f"\n📏 Y-Axis Error:")
    print(f"   MAE:  {metrics_df['y_mae'].mean():.2f} ± {metrics_df['y_mae'].std():.2f} px")
    print(f"   RMSE: {metrics_df['y_rmse'].mean():.2f} ± {metrics_df['y_rmse'].std():.2f} px")
    
    print(f"\n🎯 Coverage:")
    print(f"   Ground Truth vehicles: {df_gt['VehicleID'].nunique()}")
    print(f"   Generated vehicles: {df_generated['VehicleID'].nunique()}")
    print(f"   Successfully matched: {len(matches)}")
    print(f"   Match rate: {len(matches)/df_gt['VehicleID'].nunique()*100:.1f}%")
    
    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   • generated_trajectories.csv - Generated vehicle paths")
    print(f"   • metrics_summary.csv - All metrics")
    print(f"   • comparison_*.png - {len(matches)} comparison plots")
    
    # Step 7: Generate video overlay if requested
    if generate_video:
        video_output_path = output_dir / "vehicle_paths_overlay.mp4"
        generate_video_overlay(video_path, num_frames, roi_polygon, video_output_path, use_sahi)
        print(f"   • vehicle_paths_overlay.mp4 - Video with trajectory paths")
    
    print("="*70)

if __name__ == "__main__":
    main()
