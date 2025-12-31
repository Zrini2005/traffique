#!/usr/bin/env python3
"""
Simple vehicle path visualization - draws trajectory lines for each vehicle
"""
import cv2
import numpy as np
from collections import defaultdict
from trajectory_tracker import VehicleTrajectoryTracker

def visualize_vehicle_paths(video_path, num_frames=500, output_dir="output"):
    """Draw path lines for each vehicle as they move through frames"""
    
    # Auto-generate output path with counter
    from pathlib import Path
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    counter = 1
    while True:
        output_path = output_dir_path / f"vehicle_paths_{counter:03d}.mp4"
        if not output_path.exists():
            break
        counter += 1
    
    print("="*70)
    print("VEHICLE PATH VISUALIZATION")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print(f"Processing {num_frames} frames...")
    print(f"Output: {output_path}\n")
    
    # Setup tracker
    tracker = VehicleTrajectoryTracker(
        video_path=video_path,
        confidence_threshold=0.3,
        min_trajectory_length=5,
        roi_polygon=[(4, 873), (3827, 877), (3831, 1086), (4, 1071)]
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Colors for different vehicles (cycle through these)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 255, 0),
        (0, 128, 255), (255, 0, 128), (128, 0, 255), (0, 255, 128)
    ]
    
    # Store trajectory history
    vehicle_paths = defaultdict(list)  # vehicle_id -> list of (x, y) points
    vehicle_colors = {}  # vehicle_id -> color
    vehicle_classes = {}  # vehicle_id -> class_name
    color_idx = 0
    
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect vehicles
        result = tracker.analyzer.analyze_frame(frame, tracker.roi_polygon)
        detections = result['roi_vehicles']
        
        # Update tracker
        tracked_objects = tracker.tracker.update(detections)
        
        # Update paths for each tracked vehicle
        for vehicle_id, track_data in tracked_objects.items():
            if track_data['age'] == 0:  # Only active tracks (currently detected)
                bbox = track_data['bbox']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                
                # Add point to path
                vehicle_paths[vehicle_id].append((center_x, center_y))
                
                # Assign color if new vehicle
                if vehicle_id not in vehicle_colors:
                    vehicle_colors[vehicle_id] = colors[color_idx % len(colors)]
                    color_idx += 1
                
                # Store vehicle class for adaptive smoothing
                if vehicle_id not in vehicle_classes:
                    vehicle_classes[vehicle_id] = track_data.get('class_name', 'vehicle')
        
        # Draw all paths on frame
        for vehicle_id, path in vehicle_paths.items():
            if len(path) > 1:
                color = vehicle_colors[vehicle_id]
                
                # Apply adaptive smoothing for large vehicles to reduce jitter
                vehicle_class = vehicle_classes.get(vehicle_id, 'vehicle')
                if vehicle_class in ['truck', 'bus'] and len(path) >= 5:
                    # Light smoothing for large vehicles (running average of last 3 points)
                    smoothed_path = [path[0]]  # Keep first point
                    for i in range(1, len(path)):
                        if i < 2:
                            smoothed_path.append(path[i])
                        else:
                            # Average of last 3 points
                            avg_x = int((path[i-2][0] + path[i-1][0] + path[i][0]) / 3)
                            avg_y = int((path[i-2][1] + path[i-1][1] + path[i][1]) / 3)
                            smoothed_path.append((avg_x, avg_y))
                    draw_path = smoothed_path
                else:
                    draw_path = path
                
                # Draw path as connected line segments
                for i in range(1, len(draw_path)):
                    cv2.line(frame, draw_path[i-1], draw_path[i], color, 2, cv2.LINE_AA)
                
                # Draw current position as circle (use smoothed for large vehicles)
                current_pos = draw_path[-1] if len(draw_path) > 0 else path[-1]
                cv2.circle(frame, current_pos, 5, color, -1)
                
                # Draw vehicle ID near current position
                cv2.putText(frame, str(vehicle_id), 
                           (current_pos[0] + 10, current_pos[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add frame info
        cv2.putText(frame, f"Frame: {frame_count} | Vehicles: {len(vehicle_paths)}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"  Processed {frame_count}/{num_frames} frames | "
                  f"{len(vehicle_paths)} vehicles tracked")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Done! Video saved: {output_path}")
    print(f"\n📊 Summary:")
    print(f"   Total vehicles: {len(vehicle_paths)}")
    print(f"   Frames processed: {frame_count}")
    
    # Show top vehicles by path length
    sorted_vehicles = sorted(vehicle_paths.items(), key=lambda x: len(x[1]), reverse=True)
    print(f"\n   Top 5 longest paths:")
    for i, (vid, path) in enumerate(sorted_vehicles[:5], 1):
        print(f"     {i}. Vehicle {vid}: {len(path)} points")

if __name__ == "__main__":
    import sys
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/srini/Downloads/D2F1_stab.mp4"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    visualize_vehicle_paths(video_path, num_frames)
