#!/usr/bin/env python3
"""
Regenerate trajectories for comparison with ground truth
Focuses on accuracy improvements
"""
import sys
from pathlib import Path
import pandas as pd

# Check if we have the video
video_path = "/mnt/c/Users/srini/Downloads/D2F1_stab.mp4"
gt_path = "/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv"
output_csv = "/mnt/c/Users/srini/Downloads/trajectories_regenerated.csv"

print("="*70)
print("REGENERATING TRAJECTORIES WITH IMPROVED SETTINGS")
print("="*70)

# Load ground truth to see what frame range to process
df_gt = pd.read_csv(gt_path)
print(f"\nGround Truth Stats:")
print(f"  Frames: {df_gt['Frame'].min()} to {df_gt['Frame'].max()}")
print(f"  Vehicles: {df_gt['VehicleID'].nunique()}")
print(f"  Total points: {len(df_gt)}")

# Find the start frame (where we have ground truth)
start_frame = int(df_gt['Frame'].min())
num_frames = 200

print(f"\nWill process frames {start_frame} to {start_frame + num_frames}")
print(f"\nStarting trajectory tracking...")
print("="*70)

# Import and run tracker
from trajectory_tracker import VehicleTrajectoryTracker

# Initialize tracker with better settings
tracker = VehicleTrajectoryTracker(
    video_path=video_path,
    confidence_threshold=0.3,  # Higher confidence for better detections
    min_trajectory_length=10,
    homography_json=None,  # No homography for now
    roi_polygon=None  # Process full frame
)

# Track video
tracker.track_video(
    start_frame=start_frame,
    num_frames=num_frames
)

# Export in D2F1 format for easy comparison
csv_path = tracker.export_trajectories_csv(
    output_csv=output_csv,
    format_style="d2f1"
)

print("\n" + "="*70)
print("✅ REGENERATION COMPLETE!")
print("="*70)
print(f"📄 New CSV: {output_csv}")
print(f"\nNext step: Run comparison:")
print(f"python detailed_trajectory_comparison.py")
