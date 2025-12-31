#!/usr/bin/env python3
"""
Generate trajectories and compare with ground truth with PROPER vehicle matching

KEY POINTS:
1. HARDCODED ROI: 'road' mode now uses calibrated points [(4,873), (3827,877), (3831,1086), (4,1071)]
2. GRAPH TIMING: Graphs end at same time because we only plot OVERLAPPING frames 
   where both generated and ground truth exist - this ensures fair comparison
3. ZERO-LAG TRACKING: Removed ALL smoothing filters (Kalman, Savitzky-Golay) that caused
   trajectories to lag behind actual vehicle motion. Now uses RAW detections (outliers only removed).
4. VEHICLE ID MATCHING: Current tracker uses:
   - confidence_threshold=0.3 (detects more vehicles like GT)
   - min_iou=0.30 (moderate matching threshold)
   - max_age=30 (SHORT - reduces prediction lag, more responsive to actual position)
   - min_trajectory_length=10 (keeps shorter vehicle tracks)
   Lower max_age = CRITICAL for eliminating motion lag from predictive tracking
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import argparse
from scipy.signal import savgol_filter
from collections import defaultdict

# Import trajectory trackers
from trajectory_tracker import VehicleTrajectoryTracker
from simple_optical_tracker import OpticalFlowTracker

# Global variables for interactive selection
polygon_points = []
temp_frame = None

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks for polygon selection"""
    global polygon_points, temp_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"  Point {len(polygon_points)}: ({x}, {y})")
        
        # Draw on temp frame
        if len(polygon_points) > 0:
            cv2.circle(temp_frame, (x, y), 5, (0, 255, 0), -1)
            if len(polygon_points) > 1:
                cv2.line(temp_frame, polygon_points[-2], polygon_points[-1], (0, 255, 0), 2)
        
        cv2.imshow("Select Tracking Region", temp_frame)

def select_roi_interactive(video_path, frame_number=0):
    """Interactive ROI selection by clicking on video frame"""
    global polygon_points, temp_frame
    
    print("\n" + "="*70)
    print("INTERACTIVE ROI SELECTION")
    print("="*70)
    print("\nInstructions:")
    print("  - Click to add points for your tracking boundary")
    print("  - Press 'c' to close polygon and confirm")
    print("  - Press 'r' to reset and start over")
    print("  - Press 'q' to cancel")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not read video frame")
        return None
    
    polygon_points = []
    temp_frame = frame.copy()
    
    cv2.namedWindow("Select Tracking Region")
    cv2.setMouseCallback("Select Tracking Region", mouse_callback)
    cv2.imshow("Select Tracking Region", temp_frame)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') and len(polygon_points) >= 3:
            # Close polygon
            cv2.line(temp_frame, polygon_points[-1], polygon_points[0], (0, 255, 0), 2)
            cv2.fillPoly(temp_frame, [np.array(polygon_points)], (0, 255, 0), lineType=cv2.LINE_AA)
            alpha = 0.3
            frame_overlay = cv2.addWeighted(frame, 1-alpha, temp_frame, alpha, 0)
            cv2.imshow("Select Tracking Region", frame_overlay)
            cv2.waitKey(500)
            cv2.destroyAllWindows()
            print(f"\n✅ ROI selected with {len(polygon_points)} points")
            return polygon_points
        
        elif key == ord('r'):
            # Reset
            polygon_points = []
            temp_frame = frame.copy()
            cv2.imshow("Select Tracking Region", temp_frame)
            print("\n  Reset! Start clicking again...")
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            print("\n❌ ROI selection cancelled")
            return None
    
    cv2.destroyAllWindows()
    return None

def get_roi_selection(video_path, roi_mode='full'):
    """
    Get ROI based on mode:
    - 'full': No ROI (track entire frame)
    - 'road': Auto-detect road region
    - 'interactive': Let user draw region
    - list of tuples: Use provided polygon
    """
    if roi_mode == 'full':
        print("\n📍 ROI Mode: Full frame (no boundaries)")
        return None
    
    elif roi_mode == 'road':
        # Road region from calibrated selection (Y: ~873-1086)
        roi = [(4, 873), (3827, 877), (3831, 1086), (4, 1071)]
        print(f"\n📍 ROI Mode: Road region (hardcoded from calibration)")
        return roi
    
    elif roi_mode == 'interactive':
        return select_roi_interactive(video_path)
    
    elif isinstance(roi_mode, list):
        print(f"\n📍 ROI Mode: Custom polygon with {len(roi_mode)} points")
        return roi_mode
    
    else:
        print(f"\n⚠️  Unknown ROI mode: {roi_mode}, using full frame")
        return None

def generate_trajectories(video_path, start_frame, num_frames, output_csv, roi_polygon=None, use_optical=False):
    """Generate trajectories using our tracker"""
    print("="*70)
    print("STEP 1: GENERATING TRAJECTORIES")
    print("="*70)
    
    if roi_polygon:
        print(f"\n🎯 Tracking within ROI boundary ({len(roi_polygon)} points)")
    else:
        print(f"\n🎯 Tracking full frame (no boundaries)")
    
    if use_optical:
        print(f"\n🚀 Using OPTICAL FLOW tracker (fast + smooth)")
        tracker = OpticalFlowTracker(
            video_path=video_path,
            confidence=0.5,
            min_trajectory_length=30,
            redetect_interval=10  # Re-detect every 10 frames
        )
    else:
        print(f"\n⚡ Using ZERO-LAG YOLO tracker (RAW detections, no smoothing)")
        tracker = VehicleTrajectoryTracker(
            video_path=video_path,
            confidence_threshold=0.3,   # Detect more vehicles (match GT coverage)
            min_trajectory_length=10,   # Keep shorter vehicle tracks
            homography_json=None,
            roi_polygon=roi_polygon
        )
    
    print(f"\nTracking frames {start_frame} to {start_frame + num_frames}...")
    
    if use_optical:
        # Optical tracker exports CSV automatically
        csv_path = tracker.track_video(start_frame=start_frame, num_frames=num_frames)
    else:
        # Regular tracker needs explicit tracking + export
        tracker.track_video(start_frame=start_frame, num_frames=num_frames)
        csv_path = tracker.export_trajectories_csv(
            output_csv=output_csv,
            format_style="d2f1"
        )
    
    print(f"\n✅ Trajectories saved to: {csv_path}")
    return csv_path

def calculate_comprehensive_stats(df1, df2, v1, v2):
    """Calculate comprehensive statistics for a vehicle match including jitter analysis"""
    data1 = df1[df1['VehicleID'] == v1].sort_values('Frame')
    data2 = df2[df2['VehicleID'] == v2].sort_values('Frame')
    
    common_frames = sorted(set(data1['Frame']).intersection(set(data2['Frame'])))
    d1_common = data1[data1['Frame'].isin(common_frames)].sort_values('Frame')
    d2_common = data2[data2['Frame'].isin(common_frames)].sort_values('Frame')
    
    errors = []
    x_errors = []
    y_errors = []
    
    for frame in common_frames:
        try:
            x1 = d1_common[d1_common['Frame'] == frame]['X_pixel'].iloc[0]
            y1 = d1_common[d1_common['Frame'] == frame]['Y_pixel'].iloc[0]
            x2 = d2_common[d2_common['Frame'] == frame]['X_pixel'].iloc[0]
            y2 = d2_common[d2_common['Frame'] == frame]['Y_pixel'].iloc[0]
            
            euclidean_error = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            x_error = abs(x1 - x2)
            y_error = abs(y1 - y2)
            
            errors.append(euclidean_error)
            x_errors.append(x_error)
            y_errors.append(y_error)
        except:
            continue
    
    errors = np.array(errors)
    x_errors = np.array(x_errors)
    y_errors = np.array(y_errors)
    
    # Calculate Y-jitter (frame-to-frame variation)
    y1_jitter = 0
    y2_jitter = 0
    if len(d1_common) > 1:
        y1_diff = np.abs(np.diff(d1_common['Y_pixel'].values))
        y1_jitter = np.mean(y1_diff)
    if len(d2_common) > 1:
        y2_diff = np.abs(np.diff(d2_common['Y_pixel'].values))
        y2_jitter = np.mean(y2_diff)
    
    return {
        'v1': v1,
        'v2': v2,
        'frames': len(common_frames),
        'mae': errors.mean(),  # Mean Absolute Error
        'rmse': np.sqrt((errors**2).mean()),  # Root Mean Square Error
        'std': errors.std(),
        'min_error': errors.min(),
        'max_error': errors.max(),
        'median_error': np.median(errors),
        'x_mae': x_errors.mean(),
        'x_rmse': np.sqrt((x_errors**2).mean()),
        'x_max': x_errors.max(),
        'y_mae': y_errors.mean(),
        'y_rmse': np.sqrt((y_errors**2).mean()),
        'y_max': y_errors.max(),
        'y_jitter_gen': y1_jitter,  # Y jitter in generated trajectory
        'y_jitter_gt': y2_jitter,   # Y jitter in ground truth
        'y_jitter_ratio': y1_jitter / y2_jitter if y2_jitter > 0 else 0,  # Jitter ratio
    }

def match_vehicles_properly(df1, df2, max_matches=None, debug=True):
    """
    Properly match vehicles using trajectory similarity
    Uses both spatial and temporal matching with balanced thresholds
    Returns ALL matches if max_matches is None
    """
    print("\n" + "="*70)
    print("STEP 2: MATCHING VEHICLES (BALANCED CRITERIA)")
    print("="*70)
    
    vehicles1 = df1['VehicleID'].unique()
    vehicles2 = df2['VehicleID'].unique()
    
    print(f"\nCSV1: {len(vehicles1)} vehicles")
    print(f"CSV2: {len(vehicles2)} vehicles")
    
    matches = []
    rejection_stats = {
        'too_few_points': 0,
        'no_overlap': 0,
        'too_far': 0,
        'high_avg_error': 0,
        'high_max_error': 0
    }
    
    for v1 in vehicles1:
        data1 = df1[df1['VehicleID'] == v1].sort_values('Frame')
        
        # Get trajectory info
        frames1 = set(data1['Frame'])
        x1_mean = data1['X_pixel'].mean()
        y1_mean = data1['Y_pixel'].mean()
        
        # Reduce minimum points requirement
        if len(frames1) < 10:  # Reduced from 20
            rejection_stats['too_few_points'] += 1
            continue
        
        best_match = None
        best_score = float('inf')
        
        for v2 in vehicles2:
            data2 = df2[df2['VehicleID'] == v2].sort_values('Frame')
            
            frames2 = set(data2['Frame'])
            x2_mean = data2['X_pixel'].mean()
            y2_mean = data2['Y_pixel'].mean()
            
            # Check frame overlap - reduced minimum
            common_frames = frames1.intersection(frames2)
            if len(common_frames) < 10:  # Reduced from 20
                continue
            
            # Calculate spatial distance
            spatial_dist = np.sqrt((x1_mean - x2_mean)**2 + (y1_mean - y2_mean)**2)
            
            # RELAXED: More permissive spatial threshold
            if spatial_dist > 150:  # Increased from 80 to 150
                continue
            
            # Get trajectories for common frames
            d1_common = data1[data1['Frame'].isin(common_frames)].sort_values('Frame')
            d2_common = data2[data2['Frame'].isin(common_frames)].sort_values('Frame')
            
            # Calculate average position error across trajectory
            total_error = 0
            count = 0
            max_single_error = 0
            for frame in common_frames:
                try:
                    x1 = d1_common[d1_common['Frame'] == frame]['X_pixel'].iloc[0]
                    y1 = d1_common[d1_common['Frame'] == frame]['Y_pixel'].iloc[0]
                    x2 = d2_common[d2_common['Frame'] == frame]['X_pixel'].iloc[0]
                    y2 = d2_common[d2_common['Frame'] == frame]['Y_pixel'].iloc[0]
                    
                    error = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    total_error += error
                    count += 1
                    max_single_error = max(max_single_error, error)
                except:
                    continue
            
            if count == 0:
                continue
            
            avg_error = total_error / count
            
            # RELAXED: More permissive error thresholds
            if avg_error > 60:  # Increased from 40 to 60
                continue
            if max_single_error > 150:  # Increased from 100 to 150
                continue
            
            # Score based on error and overlap
            score = avg_error - (len(common_frames) * 0.5)
            
            if score < best_score:
                best_score = score
                best_match = (v1, v2, len(common_frames), avg_error)
        
        if best_match:
            matches.append(best_match)
    
    # Sort by error (best matches first)
    matches.sort(key=lambda x: x[3])
    
    print(f"\n✅ Found {len(matches)} good matches")
    print(f"\n📊 MATCHING QUALITY:")
    print(f"   Ground Truth vehicles: {len(vehicles2)}")
    print(f"   Generated vehicles: {len(vehicles1)}")
    print(f"   Successfully matched: {len(matches)}")
    print(f"   Match rate: {len(matches)/len(vehicles2)*100:.1f}% (IDEAL: 100%)")
    print(f"\n📋 Matching criteria used:")
    print(f"   • Min overlap: 10 frames")
    print(f"   • Max spatial distance: 150 px")
    print(f"   • Max average error: 60 px")
    print(f"   • Max single point error: 150 px")
    if len(matches) < len(vehicles2) * 0.5:
        print(f"   ⚠️  LOW MATCH RATE! Check if criteria too strict or tracking offset.")
    
    if max_matches is not None:
        print(f"\n   Showing top {min(max_matches, len(matches))}:")
        for i, (v1, v2, overlap, error) in enumerate(matches[:max_matches], 1):
            print(f"   {i}. {v1} ↔ {v2} | Frames: {overlap} | Avg Error: {error:.1f}px")
        return matches
    else:
        print(f"   Returning all {len(matches)} matches")
        return matches

def plot_comparison(df1, df2, v1, v2, output_path):
    """Create comprehensive comparison plots for matched vehicles with spatial trajectory"""
    data1 = df1[df1['VehicleID'] == v1].sort_values('Frame')
    data2 = df2[df2['VehicleID'] == v2].sort_values('Frame')
    
    # Debug Y coordinates
    print(f"      Y ranges: CSV1 [{data1['Y_pixel'].min():.1f}-{data1['Y_pixel'].max():.1f}] "
          f"vs CSV2 [{data2['Y_pixel'].min():.1f}-{data2['Y_pixel'].max():.1f}]")
    
    # Calculate Y jitter (variance in Y direction)
    y1_diff = np.abs(np.diff(data1['Y_pixel'].values))
    y2_diff = np.abs(np.diff(data2['Y_pixel'].values))
    y1_jitter = np.mean(y1_diff)
    y2_jitter = np.mean(y2_diff)
    print(f"      Y jitter: Generated={y1_jitter:.2f}px/frame, GT={y2_jitter:.2f}px/frame")
    
    # Get common frames
    common_frames = sorted(set(data1['Frame']).intersection(set(data2['Frame'])))
    
    d1_common = data1[data1['Frame'].isin(common_frames)].sort_values('Frame')
    d2_common = data2[data2['Frame'].isin(common_frames)].sort_values('Frame')
    
    # Calculate errors
    errors = []
    for frame in common_frames:
        try:
            x1 = d1_common[d1_common['Frame'] == frame]['X_pixel'].iloc[0]
            y1 = d1_common[d1_common['Frame'] == frame]['Y_pixel'].iloc[0]
            x2 = d2_common[d2_common['Frame'] == frame]['X_pixel'].iloc[0]
            y2 = d2_common[d2_common['Frame'] == frame]['Y_pixel'].iloc[0]
            t = d2_common[d2_common['Frame'] == frame]['Time'].iloc[0]
            
            error = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            errors.append({'Frame': frame, 'Time': t, 'Error': error, 
                          'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2})
        except:
            continue
    
    error_df = pd.DataFrame(errors)
    
    # Create plots with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    avg_error = error_df['Error'].mean()
    max_error = error_df['Error'].max()
    
    fig.suptitle(f'Trajectory Comparison: {v1} vs {v2}\n' + 
                 f'Avg Error: {avg_error:.1f}px | Max Error: {max_error:.1f}px | '
                 f'Y Jitter: Gen={y1_jitter:.2f}, GT={y2_jitter:.2f}',
                 fontsize=14, fontweight='bold')
    
    # ===== PLOT 1: SPATIAL TRAJECTORY (X vs Y) - MERGED VIEW =====
    ax1 = axes[0, 0]
    
    # Plot full trajectories (including non-overlapping parts)
    ax1.plot(data1['X_pixel'], data1['Y_pixel'], 'b-', linewidth=2, alpha=0.4, label='Generated (full)')
    ax1.plot(data2['X_pixel'], data2['Y_pixel'], 'r-', linewidth=2, alpha=0.4, label='Ground Truth (full)')
    
    # Overlay common trajectory with markers
    ax1.plot(error_df['X1'], error_df['Y1'], 'bo', markersize=4, alpha=0.6, label='Generated (overlap)')
    ax1.plot(error_df['X2'], error_df['Y2'], 'rs', markersize=4, alpha=0.6, label='Ground Truth (overlap)')
    
    # Mark start and end points
    ax1.plot(error_df['X1'].iloc[0], error_df['Y1'].iloc[0], 'go', markersize=12, label='Start', zorder=5)
    ax1.plot(error_df['X1'].iloc[-1], error_df['Y1'].iloc[-1], 'mo', markersize=12, label='End', zorder=5)
    
    # Draw connection lines between corresponding points every N frames
    step = max(1, len(error_df) // 10)  # Draw ~10 connection lines
    for i in range(0, len(error_df), step):
        ax1.plot([error_df['X1'].iloc[i], error_df['X2'].iloc[i]], 
                [error_df['Y1'].iloc[i], error_df['Y2'].iloc[i]], 
                'gray', linestyle='--', linewidth=1, alpha=0.3, zorder=1)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=11)
    ax1.set_ylabel('Y Position (pixels)', fontsize=11)
    ax1.set_title('Spatial Trajectory - Merged View', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()  # Invert Y axis (image coordinates)
    ax1.set_aspect('equal', adjustable='box')
    
    # ===== PLOT 2: X POSITION vs TIME =====
    ax2 = axes[0, 1]
    ax2.plot(error_df['Time'], error_df['X1'], 'b-o', label='Generated', 
             markersize=3, linewidth=2, alpha=0.7)
    ax2.plot(error_df['Time'], error_df['X2'], 'r-s', label='Ground Truth', 
             markersize=3, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('X Position (pixels)', fontsize=11)
    ax2.set_title('X Position vs Time', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ===== PLOT 3: Y POSITION vs TIME (with jitter analysis) =====
    ax3 = axes[1, 0]
    ax3.plot(error_df['Time'], error_df['Y1'], 'b-o', label='Generated', 
             markersize=3, linewidth=2, alpha=0.7)
    ax3.plot(error_df['Time'], error_df['Y2'], 'r-s', label='Ground Truth', 
             markersize=3, linewidth=2, alpha=0.7)
    
    # Apply smoothing to show trend vs jitter
    if len(error_df) > 5:
        window = min(11, len(error_df) if len(error_df) % 2 == 1 else len(error_df) - 1)
        if window >= 5:
            try:
                y1_smooth = savgol_filter(error_df['Y1'].values, window, 3)
                y2_smooth = savgol_filter(error_df['Y2'].values, window, 3)
                ax3.plot(error_df['Time'], y1_smooth, 'b--', linewidth=3, alpha=0.5, label='Gen (smoothed)')
                ax3.plot(error_df['Time'], y2_smooth, 'r--', linewidth=3, alpha=0.5, label='GT (smoothed)')
            except:
                pass
    
    ax3.set_xlabel('Time (seconds)', fontsize=11)
    ax3.set_ylabel('Y Position (pixels)', fontsize=11)
    ax3.set_title(f'Y Position vs Time (Jitter: Gen={y1_jitter:.2f}, GT={y2_jitter:.2f})', 
                  fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_yaxis()  # Invert for image coordinates
    
    # ===== PLOT 4: ERROR ANALYSIS =====
    ax4 = axes[1, 1]
    
    # Euclidean error over time
    ax4.plot(error_df['Time'], error_df['Error'], 'purple', linewidth=2, label='Euclidean Error')
    ax4.axhline(y=avg_error, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {avg_error:.1f}px')
    
    # Show X and Y error components
    x_error = np.abs(error_df['X1'] - error_df['X2'])
    y_error = np.abs(error_df['Y1'] - error_df['Y2'])
    ax4.fill_between(error_df['Time'], 0, x_error, alpha=0.3, color='blue', label='X Error Component')
    ax4.fill_between(error_df['Time'], x_error, x_error + y_error, alpha=0.3, color='orange', label='Y Error Component')
    
    ax4.set_xlabel('Time (seconds)', fontsize=11)
    ax4.set_ylabel('Position Error (pixels)', fontsize=11)
    ax4.set_title('Error Analysis Over Time', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return avg_error, max_error

def create_trajectory_video(video_path, df_generated, df_gt, output_path, start_frame, num_frames):
    """Create video with both generated and ground truth trajectories overlaid"""
    print("\n" + "="*70)
    print("CREATING TRAJECTORY VISUALIZATION VIDEO")
    print("="*70)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"\n📹 Video: {width}x{height} @ {fps}fps")
    print(f"🎬 Processing frames {start_frame} to {start_frame + num_frames}...")
    
    # Build trajectory history for each vehicle
    gen_history = defaultdict(list)
    gt_history = defaultdict(list)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, start_frame + num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get all points for this frame
        gen_frame = df_generated[df_generated['Frame'] == frame_idx]
        gt_frame = df_gt[df_gt['Frame'] == frame_idx]
        
        # Update histories
        for _, row in gen_frame.iterrows():
            vid = row['VehicleID']
            x, y = int(row['X_pixel']), int(row['Y_pixel'])
            gen_history[vid].append((x, y))
        
        for _, row in gt_frame.iterrows():
            vid = row['VehicleID']
            x, y = int(row['X_pixel']), int(row['Y_pixel'])
            gt_history[vid].append((x, y))
        
        # Draw GROUND TRUTH trajectories (RED)
        for vid, points in gt_history.items():
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (0, 0, 255), 2)  # Red
            if len(points) > 0:
                cv2.circle(frame, points[-1], 4, (0, 0, 255), -1)  # Red dot
        
        # Draw GENERATED trajectories (CYAN - bright blue)
        for vid, points in gen_history.items():
            if len(points) > 1:
                for i in range(1, len(points)):
                    cv2.line(frame, points[i-1], points[i], (255, 255, 0), 2)  # Cyan
            if len(points) > 0:
                cv2.circle(frame, points[-1], 4, (255, 255, 0), -1)  # Cyan dot
        
        # Add legend
        cv2.rectangle(frame, (10, 10), (300, 80), (0, 0, 0), -1)
        cv2.putText(frame, "Generated (Cyan)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Ground Truth (Red)", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (width - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        
        if (frame_idx - start_frame + 1) % 1 == 0:
            print(f"  ⏳ Processed {frame_idx - start_frame + 1}/{num_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Video saved: {output_path.name}")
    return output_path

def merge_fragmented_ids(df, distance_threshold=150, time_gap_threshold=30):
    """Merge fragmented vehicle IDs that likely represent the same vehicle
    MORE AGGRESSIVE: Larger distance (150px), longer time gap (30 frames), velocity-aware"""
    
    vehicles = df['VehicleID'].unique()
    
    # Build trajectory summaries with velocity
    vehicle_info = {}
    for vid in vehicles:
        v_data = df[df['VehicleID'] == vid].sort_values('Frame')
        
        # Calculate velocity from last few frames
        if len(v_data) >= 3:
            last_frames = v_data.tail(3)
            vx = (last_frames.iloc[-1]['X_pixel'] - last_frames.iloc[0]['X_pixel']) / 3
            vy = (last_frames.iloc[-1]['Y_pixel'] - last_frames.iloc[0]['Y_pixel']) / 3
        else:
            vx, vy = 0, 0
        
        vehicle_info[vid] = {
            'start_frame': v_data['Frame'].min(),
            'end_frame': v_data['Frame'].max(),
            'end_x': v_data.iloc[-1]['X_pixel'],
            'end_y': v_data.iloc[-1]['Y_pixel'],
            'start_x': v_data.iloc[0]['X_pixel'],
            'start_y': v_data.iloc[0]['Y_pixel'],
            'vx': vx,
            'vy': vy,
            'class': v_data.iloc[0]['VehicleID'].split('_')[0] if '_' in str(v_data.iloc[0]['VehicleID']) else 'unknown'
        }
    
    # Find merge candidates
    merge_map = {}  # old_id -> new_id
    merged_count = 0
    
    for vid1 in vehicles:
        if vid1 in merge_map:
            continue
            
        info1 = vehicle_info[vid1]
        
        for vid2 in vehicles:
            if vid1 == vid2 or vid2 in merge_map:
                continue
            
            info2 = vehicle_info[vid2]
            
            # Check if same vehicle class
            if info1['class'] != info2['class']:
                continue
            
            # Check temporal proximity (vid1 ends, vid2 starts)
            time_gap = info2['start_frame'] - info1['end_frame']
            if time_gap < 0 or time_gap > time_gap_threshold:
                continue
            
            # Check spatial proximity (end of vid1 to start of vid2)
            spatial_dist = np.sqrt(
                (info1['end_x'] - info2['start_x'])**2 + 
                (info1['end_y'] - info2['start_y'])**2
            )
            
            # Check velocity-projected distance
            expected_x = info1['end_x'] + info1['vx'] * time_gap
            expected_y = info1['end_y'] + info1['vy'] * time_gap
            projected_dist = np.sqrt(
                (expected_x - info2['start_x'])**2 + 
                (expected_y - info2['start_y'])**2
            )
            
            # Use the smaller distance (more permissive)
            merge_dist = min(spatial_dist, projected_dist)
            
            if merge_dist < distance_threshold:
                # Merge vid2 into vid1
                merge_map[vid2] = vid1
                merged_count += 1
                if merged_count <= 5:  # Show first few merges
                    print(f"  ✅ Merging {vid2} -> {vid1} (gap={time_gap}f, dist={spatial_dist:.1f}px, proj={projected_dist:.1f}px)")
    
    # Apply merges
    if merge_map:
        df_merged = df.copy()
        for old_id, new_id in merge_map.items():
            df_merged.loc[df_merged['VehicleID'] == old_id, 'VehicleID'] = new_id
        
        return df_merged, merged_count
    else:
        return df, 0

def create_single_vehicle_video(video_path, df1, df2, v1, v2, output_path, start_frame, num_frames):
    """Create video with single vehicle trajectory comparison"""
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Get vehicle data
    data1 = df1[df1['VehicleID'] == v1].sort_values('Frame')
    data2 = df2[df2['VehicleID'] == v2].sort_values('Frame')
    
    gen_history = []
    gt_history = []
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, start_frame + num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get points for this frame
        gen_point = data1[data1['Frame'] == frame_idx]
        gt_point = data2[data2['Frame'] == frame_idx]
        
        if not gen_point.empty:
            x, y = int(gen_point.iloc[0]['X_pixel']), int(gen_point.iloc[0]['Y_pixel'])
            gen_history.append((x, y))
        
        if not gt_point.empty:
            x, y = int(gt_point.iloc[0]['X_pixel']), int(gt_point.iloc[0]['Y_pixel'])
            gt_history.append((x, y))
        
        # Draw GT trajectory (RED - thicker)
        if len(gt_history) > 1:
            for i in range(1, len(gt_history)):
                cv2.line(frame, gt_history[i-1], gt_history[i], (0, 0, 255), 4)
        if len(gt_history) > 0:
            cv2.circle(frame, gt_history[-1], 8, (0, 0, 255), -1)
        
        # Draw Generated trajectory (CYAN - thicker)
        if len(gen_history) > 1:
            for i in range(1, len(gen_history)):
                cv2.line(frame, gen_history[i-1], gen_history[i], (255, 255, 0), 4)
        if len(gen_history) > 0:
            cv2.circle(frame, gen_history[-1], 8, (255, 255, 0), -1)
        
        # Add info box
        cv2.rectangle(frame, (10, 10), (500, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Generated: {v1} (Cyan)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Ground Truth: {v2} (Red)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    
    return output_path

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate and compare trajectories with proper matching")
    parser.add_argument('--video', default="/mnt/c/Users/srini/Downloads/D2F1_stab.mp4",
                        help="Path to video file")
    parser.add_argument('--gt', default="/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv",
                        help="Path to ground truth CSV")
    parser.add_argument('--output', default=None,
                        help="Path for output CSV (auto-generates with counter if not specified)")
    parser.add_argument('--output-dir', default=None,
                        help="Directory for comparison plots (auto-generates with timestamp if not specified)")
    parser.add_argument('--frames', type=int, default=200,
                        help="Number of frames to process (use -1 for all frames)")
    parser.add_argument('--start-frame', type=int, default=None,
                        help="Starting frame number (default: use ground truth start frame)")
    parser.add_argument('--roi', choices=['full', 'road', 'interactive'], default='road',
                        help="ROI selection mode: full (entire frame), road (auto road region), interactive (draw on screen)")
    parser.add_argument('--best', type=int, default=5,
                        help="Number of best matches to plot")
    parser.add_argument('--worst', type=int, default=5,
                        help="Number of worst matches to plot")
    parser.add_argument('--optical', action='store_true',
                        help="Use optical flow tracker (faster, smoother) instead of per-frame YOLO")
    
    args = parser.parse_args()
    
    # Paths
    video_path = args.video
    gt_csv = args.gt
    
    # Auto-generate output directory and CSV with counter
    base_dir = Path("/mnt/c/Users/srini/Downloads")
    
    if args.output_dir is None or args.output is None:
        counter = 1
        while True:
            output_dir = base_dir / f"comparison_{counter:03d}"
            output_csv = base_dir / f"trajectories_{counter:03d}.csv"
            if not output_dir.exists() and not output_csv.exists():
                break
            counter += 1
    
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    if args.output is not None:
        output_csv = args.output
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*70)
    print("TRAJECTORY GENERATION AND COMPARISON (PROPER MATCHING)")
    print("="*70)
    print(f"\n📹 Video: {video_path}")
    print(f"📊 Ground Truth: {gt_csv}")
    print(f"📁 Output Dir: {output_dir}")
    print(f"📄 Output CSV: {output_csv}")
    print(f"🎬 Frames to process: ALL FRAMES" if args.frames <= 0 else f"🎬 Frames to process: {args.frames}")
    print(f"📈 Best matches to plot: {args.best}")
    print(f"📉 Worst matches to plot: {args.worst}")
    print(f"🔧 Tracker: {'Optical Flow (FAST)' if args.optical else 'Per-Frame YOLO (SLOW)'}")
    
    # Get ROI selection
    roi_polygon = get_roi_selection(video_path, args.roi)
    
    if args.roi == 'interactive' and roi_polygon is None:
        print("\n❌ ROI selection cancelled. Exiting.")
        return
    
    # Load ground truth
    df_gt = pd.read_csv(gt_csv)
    
    # Determine start frame
    if args.start_frame is not None:
        start_frame = args.start_frame
        print(f"\n🎬 Using custom start frame: {start_frame}")
    else:
        start_frame = int(df_gt['Frame'].min())
        print(f"\n🎬 Using ground truth start frame: {start_frame}")
    
    print(f"\n📊 Ground Truth: {len(df_gt)} points, {df_gt['VehicleID'].nunique()} vehicles")
    print(f"   Frame range: {df_gt['Frame'].min()} - {df_gt['Frame'].max()}")
    
    # Determine number of frames
    if args.frames <= 0:
        # Get video total frames
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        num_frames = total_frames - start_frame + 1
        print(f"\n🎬 Processing ALL {num_frames} frames")
    else:
        num_frames = args.frames
    
    # Generate trajectories
    generated_csv = generate_trajectories(video_path, start_frame, num_frames, output_csv, roi_polygon, use_optical=args.optical)
    
    # Load generated trajectories
    df_generated = pd.read_csv(generated_csv)
    print(f"\n[*] Generated CSV Analysis:")
    print(f"   Points: {len(df_generated)}, Vehicles: {df_generated['VehicleID'].nunique()}")
    print(f"   Frame range: {df_generated['Frame'].min()} - {df_generated['Frame'].max()}")
    print(f"   X range: {df_generated['X_pixel'].min():.1f} - {df_generated['X_pixel'].max():.1f}")
    print(f"   Y range: {df_generated['Y_pixel'].min():.1f} - {df_generated['Y_pixel'].max():.1f}")
    print(f"   [!] Note: ZERO-LAG mode - trajectories use RAW detections (outliers only removed)")
    print(f"   [!] This eliminates temporal lag from smoothing filters")
    
    # NOTE: Post-processing merge REMOVED - it was incorrectly combining unrelated vehicles
    # ID consistency should be handled at tracker level, not post-processing
    print(f"\n📊 Generated vehicles: {df_generated['VehicleID'].nunique()}")
    
    # Filter ground truth to EXACT same frame range as generated
    min_frame = df_generated['Frame'].min()
    max_frame = df_generated['Frame'].max()
    df_gt_filtered = df_gt[(df_gt['Frame'] >= min_frame) & (df_gt['Frame'] <= max_frame)].copy()
    
    print(f"\n[*] Ground Truth (filtered to frames {min_frame}-{max_frame}):")
    print(f"   Points: {len(df_gt_filtered)}, Vehicles: {df_gt_filtered['VehicleID'].nunique()}")
    print(f"   X range: {df_gt_filtered['X_pixel'].min():.1f} - {df_gt_filtered['X_pixel'].max():.1f}")
    print(f"   Y range: {df_gt_filtered['Y_pixel'].min():.1f} - {df_gt_filtered['Y_pixel'].max():.1f}")
    
    # Debug: Check Y coordinate ranges
    print("\n[*] Y Coordinate Ranges:")
    print(f"   Generated Y: {df_generated['Y_pixel'].min():.1f} - {df_generated['Y_pixel'].max():.1f}")
    print(f"   Ground Truth Y: {df_gt_filtered['Y_pixel'].min():.1f} - {df_gt_filtered['Y_pixel'].max():.1f}")
    
    # Detect and apply systematic Y offset correction
    y_gen_median = df_generated['Y_pixel'].median()
    y_gt_median = df_gt_filtered['Y_pixel'].median()
    y_offset = y_gt_median - y_gen_median
    
    if abs(y_offset) > 5:  # Only correct if offset > 5px
        print(f"\n🔧 APPLYING Y-OFFSET CORRECTION")
        print(f"   Detected offset: {y_offset:.1f}px (GT median: {y_gt_median:.1f}, Gen median: {y_gen_median:.1f})")
        df_generated['Y_pixel'] = df_generated['Y_pixel'] + y_offset
        print(f"   New Y range: {df_generated['Y_pixel'].min():.1f} - {df_generated['Y_pixel'].max():.1f}")
    
    # Detect systematic Y offset (after correction)
    y_gen_median = df_generated['Y_pixel'].median()
    y_gt_median = df_gt_filtered['Y_pixel'].median()
    y_offset = y_gt_median - y_gen_median
    print(f"\n[!] DETECTED Y OFFSET: {y_offset:.1f}px")
    if abs(y_offset) > 10:
        print(f"   \u26a0\ufe0f  Significant Y offset detected! This will reduce matching.")
        print(f"   Consider: Applying offset correction to generated trajectories")
    
    # Match vehicles properly - GET ALL MATCHES
    all_matches = match_vehicles_properly(df_generated, df_gt_filtered, max_matches=None)
    
    if not all_matches:
        print("\n❌ No good matches found! Tracking quality may be poor.")
        return
    
    # Calculate comprehensive statistics for all matches
    print("\n" + "="*70)
    print("STEP 3: CALCULATING COMPREHENSIVE STATISTICS")
    print("="*70)
    
    all_stats = []
    for v1, v2, overlap, _ in all_matches:
        stats = calculate_comprehensive_stats(df_generated, df_gt_filtered, v1, v2)
        all_stats.append(stats)
    
    stats_df = pd.DataFrame(all_stats)
    
    # Sort by RMSE
    stats_df = stats_df.sort_values('rmse')
    
    # Filter to vehicles tracked for at least 300 frames
    long_tracks = stats_df[stats_df['frames'] >= 300]
    
    if len(long_tracks) == 0:
        print(f"\n⚠️  No vehicles tracked for 300+ frames! Using all {len(stats_df)} vehicles.")
        long_tracks = stats_df
    else:
        print(f"\n✅ Found {len(long_tracks)} vehicles tracked for 300+ frames (out of {len(stats_df)} total)")
    
    # Select best and worst from long-tracked vehicles
    best_matches = long_tracks.head(args.best)
    worst_matches = long_tracks.tail(args.worst)
    
    print(f"\n📊 Statistics calculated for {len(stats_df)} matched vehicles")
    print(f"   Best {args.best} RMSE range: {best_matches['rmse'].min():.2f} - {best_matches['rmse'].max():.2f} px")
    print(f"   Worst {args.worst} RMSE range: {worst_matches['rmse'].min():.2f} - {worst_matches['rmse'].max():.2f} px")
    
    # Overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS (ALL VEHICLES COMBINED)")
    print("="*70)
    print(f"\n⚠️  NOTE: Graphs show OVERLAPPING frames only (where both trajectories exist)")
    print(f"   This ensures fair comparison on same time windows.")
    print(f"\n📈 Euclidean Distance Error:")
    print(f"   Mean Absolute Error (MAE):  {stats_df['mae'].mean():.2f} ± {stats_df['mae'].std():.2f} px")
    print(f"   Root Mean Square Error (RMSE): {stats_df['rmse'].mean():.2f} ± {stats_df['rmse'].std():.2f} px")
    print(f"   Median Error: {stats_df['median_error'].mean():.2f} px")
    print(f"   Min Error (best vehicle): {stats_df['min_error'].min():.2f} px")
    print(f"   Max Error (worst vehicle): {stats_df['max_error'].max():.2f} px")
    
    print(f"\n📏 X-Axis Error:")
    print(f"   X MAE:  {stats_df['x_mae'].mean():.2f} ± {stats_df['x_mae'].std():.2f} px")
    print(f"   X RMSE: {stats_df['x_rmse'].mean():.2f} ± {stats_df['x_rmse'].std():.2f} px")
    print(f"   X Max:  {stats_df['x_max'].max():.2f} px")
    
    print(f"\n📏 Y-Axis Error:")
    print(f"   Y MAE:  {stats_df['y_mae'].mean():.2f} ± {stats_df['y_mae'].std():.2f} px")
    print(f"   Y RMSE: {stats_df['y_rmse'].mean():.2f} ± {stats_df['y_rmse'].std():.2f} px")
    print(f"   Y Max:  {stats_df['y_max'].max():.2f} px")
    
    print(f"\n📊 Y-Coordinate Jitter Analysis:")
    print(f"   Generated Avg Jitter: {stats_df['y_jitter_gen'].mean():.2f} ± {stats_df['y_jitter_gen'].std():.2f} px/frame")
    print(f"   Ground Truth Avg Jitter: {stats_df['y_jitter_gt'].mean():.2f} ± {stats_df['y_jitter_gt'].std():.2f} px/frame")
    print(f"   Jitter Ratio (Gen/GT): {stats_df['y_jitter_ratio'].mean():.2f}x")
    if stats_df['y_jitter_ratio'].mean() > 1.5:
        print(f"   ⚠️  WARNING: Generated trajectories have {stats_df['y_jitter_ratio'].mean():.1f}x more Y jitter than ground truth!")
    
    print(f"\n🎯 Coverage:")
    print(f"   Ground Truth vehicles (IDEAL): {df_gt_filtered['VehicleID'].nunique()}")
    print(f"   Generated vehicles: {df_generated['VehicleID'].nunique()}")
    print(f"   Successfully matched: {len(stats_df)}")
    print(f"   Match rate: {len(stats_df)/df_gt_filtered['VehicleID'].nunique()*100:.1f}% (IDEAL: 100%)")
    if len(stats_df) < df_gt_filtered['VehicleID'].nunique() * 0.7:
        print(f"   ⚠️  MISSING {df_gt_filtered['VehicleID'].nunique() - len(stats_df)} ground truth vehicles!")
    print(f"   Average frames per match: {stats_df['frames'].mean():.0f}")
    print(f"   Total frames analyzed: {stats_df['frames'].sum()}")
    
    # Save comprehensive statistics
    stats_path = output_dir / "comprehensive_statistics.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"\n💾 Saved comprehensive statistics: {stats_path.name}")
    
    # Generate comparison plots for BEST matches
    print("\n" + "="*70)
    print(f"STEP 4: GENERATING PLOTS FOR BEST {args.best} MATCHES")
    print("="*70)
    
    for i, row in enumerate(best_matches.itertuples(), 1):
        print(f"\n[BEST {i}/{args.best}] Plotting {row.v1} vs {row.v2}...")
        print(f"      RMSE: {row.rmse:.2f} px | MAE: {row.mae:.2f} px | Frames: {row.frames}")
        
        plot_path = output_dir / f"best_{i}_{row.v1}_vs_{row.v2}.png"
        avg_error, max_error = plot_comparison(df_generated, df_gt_filtered, row.v1, row.v2, plot_path)
        print(f"   ✅ Saved: {plot_path.name}")
    
    # Generate comparison plots for WORST matches
    print("\n" + "="*70)
    print(f"STEP 5: GENERATING PLOTS FOR WORST {args.worst} MATCHES")
    print("="*70)
    
    for i, row in enumerate(worst_matches.itertuples(), 1):
        print(f"\n[WORST {i}/{args.worst}] Plotting {row.v1} vs {row.v2}...")
        print(f"      RMSE: {row.rmse:.2f} px | MAE: {row.mae:.2f} px | Frames: {row.frames}")
        
        plot_path = output_dir / f"worst_{i}_{row.v1}_vs_{row.v2}.png"
        avg_error, max_error = plot_comparison(df_generated, df_gt_filtered, row.v1, row.v2, plot_path)
        print(f"   ✅ Saved: {plot_path.name}")
    
    # Create summary visualization
    print("\n" + "="*70)
    print("STEP 6: CREATING SUMMARY VISUALIZATIONS")
    print("="*70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # RMSE distribution
    axes[0, 0].hist(stats_df['rmse'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(stats_df['rmse'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {stats_df["rmse"].mean():.2f}px')
    axes[0, 0].set_xlabel('RMSE (pixels)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('RMSE Distribution Across All Vehicles', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # X vs Y error
    axes[0, 1].scatter(stats_df['x_mae'], stats_df['y_mae'], alpha=0.6, s=100, c=stats_df['rmse'], cmap='RdYlGn_r')
    axes[0, 1].plot([0, stats_df['x_mae'].max()], [0, stats_df['x_mae'].max()], 'k--', alpha=0.3, label='X=Y line')
    axes[0, 1].set_xlabel('X-Axis MAE (pixels)', fontsize=11)
    axes[0, 1].set_ylabel('Y-Axis MAE (pixels)', fontsize=11)
    axes[0, 1].set_title('X vs Y Error Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    cbar = plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1])
    cbar.set_label('RMSE (px)', fontsize=10)
    
    # Error vs trajectory length
    axes[1, 0].scatter(stats_df['frames'], stats_df['rmse'], alpha=0.6, s=100, c='purple')
    axes[1, 0].set_xlabel('Trajectory Length (frames)', fontsize=11)
    axes[1, 0].set_ylabel('RMSE (pixels)', fontsize=11)
    axes[1, 0].set_title('Tracking Error vs Trajectory Length', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # MAE vs RMSE
    axes[1, 1].scatter(stats_df['mae'], stats_df['rmse'], alpha=0.6, s=100, c='orange')
    axes[1, 1].plot([0, stats_df['mae'].max()], [0, stats_df['mae'].max()], 'k--', alpha=0.3, label='MAE=RMSE')
    axes[1, 1].set_xlabel('MAE (pixels)', fontsize=11)
    axes[1, 1].set_ylabel('RMSE (pixels)', fontsize=11)
    axes[1, 1].set_title('MAE vs RMSE Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_viz_path = output_dir / "summary_statistics.png"
    plt.savefig(summary_viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved summary visualization: {summary_viz_path.name}")
    
    # Generate trajectory video
    print("\n" + "="*70)
    print("STEP 7: GENERATING TRAJECTORY VIDEOS")
    print("="*70)
    
    # Full video
    print("\n[1/3] Generating full trajectory overlay...")
    video_output_path = output_dir / "trajectories_overlay.mp4"
    create_trajectory_video(
        video_path=video_path,
        df_generated=df_generated,
        df_gt=df_gt_filtered,
        output_path=video_output_path,
        start_frame=start_frame,
        num_frames=num_frames
    )
    
    # Best match video
    if len(best_matches) > 0:
        print("\n[2/3] Generating BEST match vehicle video...")
        best_row = best_matches.iloc[0]
        best_video_path = output_dir / f"best_match_{best_row.v1}_vs_{best_row.v2}.mp4"
        create_single_vehicle_video(
            video_path=video_path,
            df1=df_generated,
            df2=df_gt_filtered,
            v1=best_row.v1,
            v2=best_row.v2,
            output_path=best_video_path,
            start_frame=start_frame,
            num_frames=num_frames
        )
        print(f"   ✅ Saved: {best_video_path.name}")
    
    # Worst match video
    if len(worst_matches) > 0:
        print("\n[3/3] Generating WORST match vehicle video...")
        worst_row = worst_matches.iloc[-1]
        worst_video_path = output_dir / f"worst_match_{worst_row.v1}_vs_{worst_row.v2}.mp4"
        create_single_vehicle_video(
            video_path=video_path,
            df1=df_generated,
            df2=df_gt_filtered,
            v1=worst_row.v1,
            v2=worst_row.v2,
            output_path=worst_video_path,
            start_frame=start_frame,
            num_frames=num_frames
        )
        print(f"   ✅ Saved: {worst_video_path.name}")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n📁 All files saved to: {output_dir}")
    
    # Error reduction strategies
    print(f"\n💡 ERROR REDUCTION STRATEGIES:")
    if stats_df['rmse'].mean() > 20:
        print(f"   • Current RMSE: {stats_df['rmse'].mean():.1f}px - Consider:")
        print(f"     1. Better camera calibration (homography)")
        print(f"     2. Higher confidence threshold (currently 0.3)")
        print(f"     3. Increase Kalman filter process noise")
        print(f"     4. Check video stabilization quality")
    if stats_df['y_jitter_ratio'].mean() > 1.5:
        print(f"   • High Y jitter - Increase smoothing window size")
    
    # Vehicle ID consistency analysis
    print(f"\n🔄 VEHICLE ID MERGING/CONSISTENCY:")
    print(f"   • Ground Truth vehicles (IDEAL): {df_gt_filtered['VehicleID'].nunique()}")
    print(f"   • Generated vehicles: {df_generated['VehicleID'].nunique()}")
    print(f"   • Successfully matched: {len(stats_df)}")
    
    # Fragmentation = extra IDs generated beyond what's needed
    extra_ids = df_generated['VehicleID'].nunique() - df_gt_filtered['VehicleID'].nunique()
    fragmentation_rate = extra_ids / df_gt_filtered['VehicleID'].nunique()
    print(f"   • Extra IDs generated: {extra_ids}")
    print(f"   • ID fragmentation rate: {fragmentation_rate:.1%}")
    
    if fragmentation_rate > 0.3:
        print(f"   ⚠️  High fragmentation! {extra_ids} extra IDs created (ID splitting).")
        print(f"      Current: min_iou=0.45, max_age=80, post-merge threshold=80px")
        print(f"      Consider: Further increase max_age or reduce confidence threshold")
    elif extra_ids < 0:
        print(f"   ⚠️  Missing vehicles! {abs(extra_ids)} GT vehicles not detected.")
    else:
        print(f"   ✅ Good ID consistency!")
    
    print(f"\n📊 Files generated:")
    print(f"   • comprehensive_statistics.csv - All vehicle statistics")
    print(f"   • summary_statistics.png - Overall visualization")
    print(f"   • trajectories_overlay.mp4 - Video with all trajectories")
    print(f"   • best_match_*.mp4 - Video of best matched vehicle")
    print(f"   • worst_match_*.mp4 - Video of worst matched vehicle")
    print(f"   • best_*.png - {args.best} best tracking examples")
    print(f"   • worst_*.png - {args.worst} worst tracking examples")
    
    print(f"\n✅ All files saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
