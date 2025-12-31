#!/usr/bin/env python3
"""
Create video overlay from trajectory CSV
Reads pre-generated trajectories and draws them on video
"""
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

def create_overlay_from_csv(csv_path, video_path, output_path=None, max_frames=None):
    """Create video with trajectory overlays from CSV file"""
    
    print("="*70)
    print("VIDEO OVERLAY FROM CSV")
    print("="*70)
    print(f"\nCSV: {csv_path}")
    print(f"Video: {video_path}")
    
    # Read trajectories from CSV
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} trajectory points for {df['VehicleID'].nunique()} vehicles")
    
    # Auto-generate output path if not provided
    if output_path is None:
        csv_dir = Path(csv_path).parent
        output_path = csv_dir / "trajectory_overlay.mp4"
    
    print(f"Output: {output_path}\n")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame range
    max_csv_frame = int(df['Frame'].max())
    if max_frames is None:
        max_frames = min(max_csv_frame + 1, total_frames)
    
    print(f"Processing {max_frames} frames at {fps} fps...\n")
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Colors for different vehicles
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
    ]
    
    # Organize trajectories by vehicle and frame
    vehicle_trajectories = defaultdict(list)
    for _, row in df.iterrows():
        vehicle_id = int(row['VehicleID'])
        frame = int(row['Frame'])
        x = int(row['X_pixel'])
        y = int(row['Y_pixel'])
        vehicle_trajectories[vehicle_id].append((frame, x, y))
    
    # Sort each vehicle's trajectory by frame
    for vehicle_id in vehicle_trajectories:
        vehicle_trajectories[vehicle_id].sort(key=lambda x: x[0])
    
    # Assign colors to vehicles
    vehicle_colors = {}
    for i, vehicle_id in enumerate(sorted(vehicle_trajectories.keys())):
        vehicle_colors[vehicle_id] = colors[i % len(colors)]
    
    # Store accumulated paths up to current frame
    accumulated_paths = defaultdict(list)
    
    frame_count = 0
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update accumulated paths for this frame
        for vehicle_id, trajectory in vehicle_trajectories.items():
            for traj_frame, x, y in trajectory:
                if traj_frame == frame_count:
                    accumulated_paths[vehicle_id].append((x, y))
        
        # Draw all accumulated paths
        active_vehicles = 0
        for vehicle_id, path in accumulated_paths.items():
            if len(path) > 0:
                color = vehicle_colors[vehicle_id]
                
                # Draw path lines
                for i in range(1, len(path)):
                    cv2.line(frame, path[i-1], path[i], color, 2, cv2.LINE_AA)
                
                # Check if vehicle is active in this frame (has point at this frame)
                is_active = any(f == frame_count for f, _, _ in vehicle_trajectories[vehicle_id])
                
                if is_active:
                    active_vehicles += 1
                    # Draw current position
                    current_pos = path[-1]
                    cv2.circle(frame, current_pos, 5, color, -1)
                    
                    # Draw vehicle ID
                    cv2.putText(frame, str(vehicle_id), 
                               (current_pos[0] + 10, current_pos[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count} | Active: {active_vehicles} | Total: {len(accumulated_paths)}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Processed {frame_count}/{max_frames} frames...")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Done! Video saved: {output_path}")
    print(f"\n📊 Summary:")
    print(f"   Total vehicles: {len(accumulated_paths)}")
    print(f"   Frames processed: {frame_count}")
    print("="*70)

if __name__ == "__main__":
    import sys
    
    # Hardcoded CSV path
    csv_path = r"\\wsl.localhost\Ubuntu-24.04\home\zrini\traffique\output\comparison_004\generated_trajectories.csv"
    
    # Convert Windows path to WSL path
    if csv_path.startswith(r"\\wsl.localhost\Ubuntu-24.04"):
        csv_path = csv_path.replace(r"\\wsl.localhost\Ubuntu-24.04", "")
        csv_path = csv_path.replace("\\", "/")
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/srini/Downloads/D2F1_stab.mp4"
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    create_overlay_from_csv(csv_path, video_path, max_frames=max_frames)
