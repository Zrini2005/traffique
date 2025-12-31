#!/usr/bin/env python3
"""
Generate individual vehicle path videos - shows 3 random vehicles with their complete paths
"""
import cv2
import numpy as np
from collections import defaultdict
import random
from trajectory_tracker import VehicleTrajectoryTracker

def generate_individual_vehicle_videos(video_path, num_frames=500, output_dir="output"):
    """Generate videos for 3 random vehicles showing their individual paths"""
    
    print("="*70)
    print("INDIVIDUAL VEHICLE PATH VISUALIZATION")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print(f"Processing {num_frames} frames to collect vehicle data...\n")
    
    # First pass: Track all vehicles to select longest 3
    tracker = VehicleTrajectoryTracker(
        video_path=video_path,
        confidence_threshold=0.3,
        min_trajectory_length=5,
        roi_polygon=[(4, 873), (3827, 877), (3831, 1086), (4, 1071)]
    )
    
    cap = cv2.VideoCapture(video_path)
    
    # Store all vehicle paths frame by frame
    vehicle_paths = defaultdict(list)  # vehicle_id -> list of (frame_num, x, y)
    frame_count = 0
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        result = tracker.analyzer.analyze_frame(frame, tracker.roi_polygon)
        detections = result['roi_vehicles']
        tracked_objects = tracker.tracker.update(detections)
        
        for vehicle_id, track_data in tracked_objects.items():
            if track_data['age'] == 0:
                bbox = track_data['bbox']
                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                vehicle_paths[vehicle_id].append((frame_count, center_x, center_y))
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Collecting data: {frame_count}/{num_frames} frames...")
    
    cap.release()
    
    # Select 3 vehicles with longest paths
    sorted_vehicles = sorted(vehicle_paths.items(), key=lambda x: len(x[1]), reverse=True)
    selected_vehicles = sorted_vehicles[:3] if len(sorted_vehicles) >= 3 else sorted_vehicles
    
    print(f"\n📊 Selected 3 vehicles for individual videos:")
    for i, (vid, path) in enumerate(selected_vehicles, 1):
        print(f"  {i}. Vehicle {vid}: {len(path)} points")
    
    # Generate individual videos for each selected vehicle
    for vehicle_id, path in selected_vehicles:
        generate_single_vehicle_video(video_path, vehicle_id, path, num_frames, output_dir)
    
    print(f"\n✅ All videos generated in {output_dir}/")

def generate_single_vehicle_video(video_path, vehicle_id, path, num_frames, output_dir):
    """Generate video for a single vehicle with its path highlighted"""
    
    from pathlib import Path
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(exist_ok=True, parents=True)
    
    # Find next available counter
    counter = 1
    while True:
        output_path = output_dir_path / f"vehicle_{vehicle_id}_{counter:03d}.mp4"
        if not output_path.exists():
            break
        counter += 1
    
    print(f"\n🎬 Generating video for Vehicle {vehicle_id}...")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Convert path to dict for fast lookup
    path_dict = {frame_num: (x, y) for frame_num, x, y in path}
    
    # Collect all path points up to current frame
    current_path = []
    
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if vehicle is in this frame
        if frame_count in path_dict:
            current_path.append(path_dict[frame_count])
        
        # Draw accumulated path
        if len(current_path) > 1:
            # Draw path line
            for i in range(1, len(current_path)):
                # Gradient effect - older points dimmer
                alpha = 0.3 + 0.7 * (i / len(current_path))
                color = (0, int(255 * alpha), int(255 * alpha))  # Cyan gradient
                cv2.line(frame, current_path[i-1], current_path[i], color, 3, cv2.LINE_AA)
            
            # Draw current position (bright)
            cv2.circle(frame, current_path[-1], 8, (0, 255, 255), -1)
            cv2.circle(frame, current_path[-1], 10, (255, 255, 255), 2)
        
        # Add info overlay
        cv2.rectangle(frame, (10, 10), (500, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Vehicle ID: {vehicle_id}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}/{num_frames}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if len(current_path) > 0:
            cv2.putText(frame, f"Path length: {len(current_path)} points", (20, 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    
    print(f"   ✅ Saved: {output_path}")

if __name__ == "__main__":
    import sys
    
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/srini/Downloads/D2F1_stab.mp4"
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    generate_individual_vehicle_videos(video_path, num_frames)
