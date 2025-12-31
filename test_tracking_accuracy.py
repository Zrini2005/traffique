"""
Test tracking accuracy by visualizing raw vs smoothed detections
This helps identify if the issue is in detection, tracking, or smoothing
"""

import cv2
import numpy as np
from pathlib import Path
from trajectory_tracker import VehicleTrajectoryTracker

def test_tracking_accuracy(
    video_path: str = "data/videos/video1.mp4",
    roi_polygon=None,
    start_frame: int = 1000,
    end_frame: int = 1200
):
    """
    Generate side-by-side comparison of:
    1. Raw detections (bounding boxes)
    2. Raw trajectory points (unsmoothed centers)
    3. Smoothed trajectory points
    """
    
    if roi_polygon is None:
        roi_polygon = [(4,873), (3827,877), (3831,1086), (4,1071)]
    
    print("\n" + "="*80)
    print("🔍 TRACKING ACCURACY TEST")
    print("="*80)
    print(f"Video: {video_path}")
    print(f"Frames: {start_frame} to {end_frame}")
    print(f"ROI: {roi_polygon}")
    
    # Initialize tracker
    tracker = VehicleTrajectoryTracker(
        video_path=video_path,
        roi_polygon=roi_polygon
    )
    
    # Track video
    tracker.track_video(start_frame=start_frame, end_frame=end_frame)
    
    # Create output directory
    output_dir = Path("output/accuracy_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate comparison video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = output_dir / "tracking_accuracy_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"\n📹 Generating accuracy test video...")
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        vis_frame = frame.copy()
        
        # Draw ROI polygon
        if roi_polygon:
            pts = np.array(roi_polygon, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis_frame, [pts], True, (255, 255, 0), 2)
        
        # For each vehicle, draw:
        # 1. Current raw position (if available) - GREEN circle
        # 2. Raw trajectory history - YELLOW line
        # 3. Smoothed trajectory - CYAN line
        
        for vehicle_id in tracker.raw_trajectories.keys():
            frames = tracker.frame_indices[vehicle_id]
            raw_traj = tracker.raw_trajectories[vehicle_id]
            
            # Draw raw trajectory up to this frame
            for i, f in enumerate(frames):
                if f > frame_idx:
                    break
                if f < start_frame:
                    continue
                
                pos = raw_traj[i]
                pt = (int(pos[0]), int(pos[1]))
                
                # Current position - large green circle
                if f == frame_idx:
                    cv2.circle(vis_frame, pt, 8, (0, 255, 0), -1)
                    cv2.circle(vis_frame, pt, 10, (255, 255, 255), 2)
                    
                    # Label
                    cv2.putText(vis_frame, f"ID:{vehicle_id}", 
                               (pt[0]+15, pt[1]-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Connect with previous point - yellow line (raw)
                if i > 0 and frames[i-1] <= frame_idx:
                    prev_pos = raw_traj[i-1]
                    prev_pt = (int(prev_pos[0]), int(prev_pos[1]))
                    cv2.line(vis_frame, prev_pt, pt, (0, 255, 255), 2)
            
            # Draw smoothed trajectory if available
            if vehicle_id in tracker.smooth_trajectories:
                smooth_traj = tracker.smooth_trajectories[vehicle_id]
                
                for i, f in enumerate(frames):
                    if f > frame_idx:
                        break
                    if f < start_frame or i >= len(smooth_traj):
                        continue
                    
                    pos = smooth_traj[i]
                    pt = (int(pos[0]), int(pos[1]))
                    
                    # Connect with previous point - cyan line (smoothed)
                    if i > 0 and frames[i-1] <= frame_idx and (i-1) < len(smooth_traj):
                        prev_pos = smooth_traj[i-1]
                        prev_pt = (int(prev_pos[0]), int(prev_pos[1]))
                        cv2.line(vis_frame, prev_pt, pt, (255, 255, 0), 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_frame, "Green Circle = Current Detection Center", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        legend_y += 30
        cv2.putText(vis_frame, "Yellow Line = Raw Trajectory (unsmoothed)", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        legend_y += 30
        cv2.putText(vis_frame, "Cyan Line = Smoothed Trajectory", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        legend_y += 40
        cv2.putText(vis_frame, f"Frame: {frame_idx}", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(vis_frame)
        
        if (frame_idx - start_frame) % 50 == 0:
            print(f"  Processed {frame_idx - start_frame}/{end_frame - start_frame} frames")
    
    cap.release()
    out.release()
    
    print(f"\n✅ Video saved to: {output_path}")
    print("\nAnalysis:")
    print("- If GREEN circles are NOT on vehicle centers → Detection/bbox issue")
    print("- If YELLOW line is accurate but CYAN is off → Smoothing too aggressive")
    print("- If both are off → Detection model needs tuning")
    
    # Print statistics
    print(f"\n📊 Statistics:")
    print(f"Total vehicles tracked: {len(tracker.raw_trajectories)}")
    print(f"Vehicles with smooth trajectories: {len(tracker.smooth_trajectories)}")
    
    # Check fragmentation
    gt_csv = "data/groundtruth/video1_groundtruth.csv"
    if Path(gt_csv).exists():
        import pandas as pd
        gt_df = pd.read_csv(gt_csv)
        unique_gt = gt_df['vehicle_id'].nunique()
        tracked = len(tracker.raw_trajectories)
        frag_rate = (tracked - unique_gt) / unique_gt * 100
        print(f"Ground truth vehicles: {unique_gt}")
        print(f"Fragmentation rate: {frag_rate:.1f}%")

if __name__ == "__main__":
    test_tracking_accuracy()
