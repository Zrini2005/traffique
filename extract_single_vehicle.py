#!/usr/bin/env python3
"""
Extract a single vehicle trajectory for ground truth validation.
Shows the vehicle in each frame so you can annotate the same one.

Usage:
  python extract_single_vehicle.py --csv <trajectory_csv> --vehicle-id <id> --frames <frame_list> --video <video_path> --output <output_dir>
"""

import pandas as pd
import cv2
import argparse
from pathlib import Path
import numpy as np


def extract_and_visualize(csv_path, vehicle_id, frames, video_path, output_dir):
    """Extract one vehicle's trajectory and create visualization frames"""
    
    # Load trajectories
    df = pd.read_csv(csv_path)
    
    # Filter for this vehicle
    vehicle_df = df[df['vehicle_id'] == vehicle_id].copy()
    
    if len(vehicle_df) == 0:
        print(f"❌ Vehicle {vehicle_id} not found in CSV")
        return
    
    print(f"✓ Found vehicle {vehicle_id} with {len(vehicle_df)} trajectory points")
    print(f"  Frame range: {vehicle_df['frame'].min()} - {vehicle_df['frame'].max()}")
    
    # Filter for requested frames
    vehicle_subset = vehicle_df[vehicle_df['frame'].isin(frames)]
    
    if len(vehicle_subset) == 0:
        print(f"❌ Vehicle {vehicle_id} not present in requested frames")
        print(f"Available frames for this vehicle: {sorted(vehicle_df['frame'].unique())[:20]}...")
        return
    
    print(f"✓ Vehicle present in {len(vehicle_subset)} requested frames")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save prediction CSV
    pred_csv = output_path / f"vehicle_{vehicle_id}_predictions.csv"
    vehicle_subset.to_csv(pred_csv, index=False)
    print(f"✓ Saved predictions: {pred_csv}")
    
    # Create template ground truth CSV
    gt_template = vehicle_subset[['vehicle_id', 'frame', 'x_px', 'y_px']].copy()
    gt_template['x_px'] = 0  # User will fill these
    gt_template['y_px'] = 0
    gt_csv = output_path / f"vehicle_{vehicle_id}_ground_truth_template.csv"
    gt_template.to_csv(gt_csv, index=False)
    print(f"✓ Saved GT template: {gt_csv}")
    
    # Create annotated video frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    print(f"\n📹 Creating annotated frames...")
    
    for _, row in vehicle_subset.iterrows():
        frame_no = int(row['frame'])
        x_px = float(row['x_px'])
        y_px = float(row['y_px'])
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        
        if not ret:
            print(f"  ⚠️  Failed to read frame {frame_no}")
            continue
        
        # Draw large marker on vehicle
        center = (int(x_px), int(y_px))
        
        # Red circle
        cv2.circle(frame, center, 25, (0, 0, 255), 3)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        # Crosshair
        cv2.line(frame, (center[0]-30, center[1]), (center[0]+30, center[1]), (0, 255, 0), 2)
        cv2.line(frame, (center[0], center[1]-30), (center[0], center[1]+30), (0, 255, 0), 2)
        
        # Label
        label = f"V{vehicle_id} - Frame {frame_no}"
        cv2.putText(frame, label, (center[0]+40, center[1]-40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        # Instructions
        cv2.putText(frame, f"Click on the CENTER of this vehicle", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Predicted position marked with RED circle", (50, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Save frame
        output_frame = output_path / f"frame_{frame_no:06d}_vehicle_{vehicle_id}.png"
        cv2.imwrite(str(output_frame), frame)
        print(f"  ✓ Frame {frame_no}")
    
    cap.release()
    
    print(f"\n✅ Done! Next steps:")
    print(f"\n1. Run the annotation tool on these specific frames:")
    print(f"   python annotate_single_vehicle.py --frames {','.join(map(str, frames))} --vehicle-id {vehicle_id} --video \"{video_path}\" --output \"{output_path / f'vehicle_{vehicle_id}_ground_truth.csv'}\"")
    print(f"\n2. Or manually edit: {gt_csv}")
    print(f"   (Look at the images in {output_dir} to see where to click)")
    print(f"\n3. Run metrics:")
    print(f"   python trajectory_accuracy_metrics.py --csv \"{pred_csv}\" --gt \"<your_annotations.csv>\" --output \"{output_path}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Extract single vehicle trajectory for ground truth validation"
    )
    parser.add_argument("--csv", required=True, help="Full trajectory CSV")
    parser.add_argument("--vehicle-id", type=int, required=True, help="Vehicle ID to extract")
    parser.add_argument("--frames", required=True, help="Comma-separated frame numbers")
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--output", default="output/single_vehicle_validation", help="Output directory")
    
    args = parser.parse_args()
    
    frames = [int(f.strip()) for f in args.frames.split(',')]
    
    extract_and_visualize(args.csv, args.vehicle_id, frames, args.video, args.output)


if __name__ == "__main__":
    main()
