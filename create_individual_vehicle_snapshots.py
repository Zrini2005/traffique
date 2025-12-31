#!/usr/bin/env python3
"""
Create individual snapshots for each vehicle from two CSVs

Takes 10 random vehicles from each CSV and creates a separate image for each,
showing the vehicle position on the video frame.

Usage:
  python create_individual_vehicle_snapshots.py --csv1 csv1.csv --csv2 csv2.csv --video video.mp4 --output output_folder
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


class VehicleSnapshotCreator:
    """Create individual snapshots for vehicles from CSVs"""
    
    def __init__(self, csv1_path: str, csv2_path: str, video_path: str):
        """Load CSVs and video"""
        self.df1 = pd.read_csv(csv1_path)
        self.df2 = pd.read_csv(csv2_path)
        self.video_path = video_path
        self.video_cap = None
        
        # Normalize column names
        self.df1 = self._normalize_columns(self.df1, "CSV1")
        self.df2 = self._normalize_columns(self.df2, "CSV2")
        
        print(f"✓ Loaded CSV1: {len(self.df1)} points, {self.df1['vehicle_id'].nunique()} vehicles")
        print(f"✓ Loaded CSV2: {len(self.df2)} points, {self.df2['vehicle_id'].nunique()} vehicles")
    
    def _normalize_columns(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        """Normalize column names"""
        df = df.copy()
        
        # Detect format
        if 'X_pixel' in df.columns:
            df.rename(columns={
                'Frame': 'frame',
                'VehicleID': 'vehicle_id',
                'X_pixel': 'x_px',
                'Y_pixel': 'y_px',
                'Class': 'class',
                'Time': 'time'
            }, inplace=True)
        
        df['source'] = label
        return df
    
    def get_video_frame(self, frame_number: int):
        """Extract a frame from the video"""
        if self.video_cap is None:
            self.video_cap = cv2.VideoCapture(self.video_path)
        
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video_cap.read()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def sample_random_vehicles(self, df: pd.DataFrame, n_samples: int = 10):
        """Sample N random vehicles from dataframe"""
        all_vehicles = df['vehicle_id'].unique()
        sampled = np.random.choice(all_vehicles, min(n_samples, len(all_vehicles)), replace=False)
        return sampled
    
    def create_vehicle_snapshot(self, df: pd.DataFrame, vehicle_id: str, output_path: Path, csv_label: str, index: int):
        """Create a snapshot for one vehicle"""
        
        # Get vehicle data
        vehicle_data = df[df['vehicle_id'] == vehicle_id].sort_values('frame')
        
        if len(vehicle_data) == 0:
            print(f"  ⚠️  No data for {vehicle_id}")
            return
        
        # Pick a random row from this vehicle's trajectory
        random_row = vehicle_data.sample(n=1).iloc[0]
        
        frame_num = int(random_row['frame'])
        x_px = random_row['x_px']
        y_px = random_row['y_px']
        vclass = random_row.get('class', 'Unknown')
        time = random_row.get('time', frame_num * 0.04)  # Assume 25fps if no time
        
        # Get video frame
        try:
            frame = self.get_video_frame(frame_num)
        except Exception as e:
            print(f"  ⚠️  Could not get frame {frame_num}: {e}")
            return
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.imshow(frame)
        
        # Plot vehicle position
        ax.scatter(x_px, y_px, c='red', s=30, marker='o', 
                  edgecolors='yellow', linewidths=2, zorder=10)
        
        # Add crosshair
        ax.axhline(y=y_px, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
        ax.axvline(x=x_px, color='yellow', linestyle='--', alpha=0.5, linewidth=1)
        
        # Add label box
        info_text = (f"{csv_label} - Vehicle #{index}\n"
                    f"ID: {vehicle_id}\n"
                    f"Class: {vclass}\n"
                    f"Frame: {frame_num}\n"
                    f"Time: {time:.2f}s\n"
                    f"Position: ({x_px:.1f}, {y_px:.1f})")
        
        ax.text(0.02, 0.98, info_text, 
               transform=ax.transAxes, 
               fontsize=11, 
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
               color='white',
               family='monospace')
        
        ax.axis('off')
        ax.set_title(f'{csv_label}: {vehicle_id} ({vclass})', fontsize=14, fontweight='bold', pad=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ {output_path.name}: {vehicle_id} ({vclass}) at frame {frame_num} ({x_px:.0f}, {y_px:.0f})")
    
    def create_all_snapshots(self, output_dir: str, n_samples: int = 10):
        """Create snapshots for sampled vehicles from both CSVs"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sample vehicles
        print(f"\n📸 Sampling {n_samples} vehicles from each CSV...\n")
        vehicles1 = self.sample_random_vehicles(self.df1, n_samples)
        vehicles2 = self.sample_random_vehicles(self.df2, n_samples)
        
        # Create CSV1 snapshots
        print(f"Creating CSV1 snapshots:")
        for i, vid in enumerate(vehicles1, 1):
            snapshot_path = output_path / f"CSV1_{i:02d}_{vid}.png"
            self.create_vehicle_snapshot(self.df1, vid, snapshot_path, "CSV1", i)
        
        print(f"\nCreating CSV2 snapshots:")
        # Create CSV2 snapshots
        for i, vid in enumerate(vehicles2, 1):
            snapshot_path = output_path / f"CSV2_{i:02d}_{vid}.png"
            self.create_vehicle_snapshot(self.df2, vid, snapshot_path, "CSV2", i)
        
        # Close video
        if self.video_cap is not None:
            self.video_cap.release()
        
        print(f"\n✅ Created {n_samples * 2} snapshots in: {output_dir}")
        print(f"   • {n_samples} from CSV1")
        print(f"   • {n_samples} from CSV2")


def main():
    parser = argparse.ArgumentParser(
        description="Create individual vehicle snapshots from two CSVs"
    )
    parser.add_argument("--csv1", required=True, help="First trajectory CSV")
    parser.add_argument("--csv2", required=True, help="Second trajectory CSV")
    parser.add_argument("--video", required=True, help="Video file")
    parser.add_argument("--output", default="output/vehicle_snapshots", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of vehicles per CSV")
    
    args = parser.parse_args()
    
    creator = VehicleSnapshotCreator(args.csv1, args.csv2, args.video)
    creator.create_all_snapshots(args.output, args.n_samples)


if __name__ == "__main__":
    main()
