#!/usr/bin/env python3
"""
Debug tracking issues - shows what's happening frame by frame
"""
import sys
from trajectory_tracker import VehicleTrajectoryTracker

def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/c/Users/srini/Downloads/D2F1_stab.mp4"
    
    print("="*70)
    print("TRACKING DEBUG MODE - Detailed frame-by-frame analysis")
    print("="*70)
    print(f"\nVideo: {video_path}")
    print("Processing first 50 frames with detailed output...\n")
    
    # Create tracker with debug output enabled
    tracker = VehicleTrajectoryTracker(
        video_path=video_path,
        confidence_threshold=0.2,
        min_trajectory_length=10,  # Low threshold to see all tracks
        roi_polygon=[(4, 873), (3827, 877), (3831, 1086), (4, 1071)]  # Road ROI
    )
    
    # Track with detailed output
    tracker.track_video(start_frame=0, num_frames=1000)
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total vehicles tracked: {len(tracker.raw_trajectories)}")
    print(f"\nTop 10 vehicles by trajectory length:")
    
    sorted_tracks = sorted(tracker.raw_trajectories.items(), 
                          key=lambda x: len(x[1]), reverse=True)
    
    for i, (vid, traj) in enumerate(sorted_tracks[:10], 1):
        frames = tracker.frame_indices[vid]
        print(f"  {i}. Vehicle {vid}: {len(traj)} points "
              f"(frames {min(frames)}-{max(frames)})")
    
    print("\n💡 Look for:")
    print("   • Low detection counts = increase confidence or check ROI")
    print("   • Many 'lost' detections = IoU threshold too high (currently 0.30)")
    print("   • Many new tracks = ID fragmentation, tracks dying too fast")
    print("   • High 'aged tracks' = vehicles leaving frame but tracks persisting")

if __name__ == "__main__":
    main()
