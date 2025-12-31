#!/usr/bin/env python3
"""
Visualize Sample Points from Two CSV Files on Video Frame

Shows 10 vehicles from each CSV plotted on the video to help manually identify matches.

Usage:
  python visualize_trajectory_points.py --csv1 csv1.csv --csv2 csv2.csv --video video.mp4 --output output.png
"""

import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
import random


class TrajectoryVisualizer:
    """Visualize sample points from two tracking CSVs on video"""
    
    def __init__(self, csv1_path: str, csv2_path: str, video_path: str):
        """Load CSVs and video"""
        self.df1 = pd.read_csv(csv1_path)
        self.df2 = pd.read_csv(csv2_path)
        self.video_path = video_path
        
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
                'Class': 'class'
            }, inplace=True)
        
        df['source'] = label
        return df
    
    def get_video_frame(self, frame_number: int = None):
        """Extract a frame from the video"""
        cap = cv2.VideoCapture(self.video_path)
        
        if frame_number is None:
            # Get middle frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_number = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not read frame {frame_number} from video")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"✓ Extracted frame {frame_number}, resolution: {frame.shape[1]}x{frame.shape[0]}")
        return frame, frame_number
    
    def sample_vehicles(self, df: pd.DataFrame, n_samples: int = 10, frame_num: int = None):
        """
        Sample N vehicles from the dataframe
        
        If frame_num is specified, prefer vehicles visible in that frame
        """
        if frame_num is not None:
            # Get vehicles visible in target frame ±100 frames
            frame_window = df[(df['frame'] >= frame_num - 100) & (df['frame'] <= frame_num + 100)]
            vehicles_in_window = frame_window['vehicle_id'].unique()
            
            if len(vehicles_in_window) >= n_samples:
                sampled = np.random.choice(vehicles_in_window, n_samples, replace=False)
            else:
                # Not enough, add random ones
                all_vehicles = df['vehicle_id'].unique()
                remaining = set(all_vehicles) - set(vehicles_in_window)
                additional = np.random.choice(list(remaining), 
                                             min(n_samples - len(vehicles_in_window), len(remaining)), 
                                             replace=False)
                sampled = list(vehicles_in_window) + list(additional)
        else:
            # Random sample
            all_vehicles = df['vehicle_id'].unique()
            sampled = np.random.choice(all_vehicles, min(n_samples, len(all_vehicles)), replace=False)
        
        return sampled[:n_samples]
    
    def get_vehicle_position_at_frame(self, df: pd.DataFrame, vehicle_id: str, target_frame: int):
        """Get vehicle position closest to target frame"""
        vehicle_data = df[df['vehicle_id'] == vehicle_id]
        
        # Find closest frame
        closest_idx = (vehicle_data['frame'] - target_frame).abs().idxmin()
        row = vehicle_data.loc[closest_idx]
        
        return row['x_px'], row['y_px'], row['frame'], row.get('class', 'Unknown')
    
    def visualize(self, n_samples: int = 10, output_path: str = "output/trajectory_visualization.png"):
        """Create visualization with labeled points from both CSVs"""
        
        # Get video frame
        frame, frame_num = self.get_video_frame()
        
        # Sample vehicles from both CSVs
        print(f"\n📍 Sampling {n_samples} vehicles from each CSV...")
        vehicles1 = self.sample_vehicles(self.df1, n_samples, frame_num)
        vehicles2 = self.sample_vehicles(self.df2, n_samples, frame_num)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        # Colors for labeling
        colors1 = plt.cm.Set1(np.linspace(0, 1, n_samples))
        colors2 = plt.cm.Set2(np.linspace(0, 1, n_samples))
        
        # Plot CSV1 on left
        axes[0].imshow(frame)
        axes[0].set_title(f'CSV1 Vehicles (Frame {frame_num})', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        print(f"\nCSV1 vehicles:")
        for i, vid in enumerate(vehicles1):
            x, y, actual_frame, vclass = self.get_vehicle_position_at_frame(self.df1, vid, frame_num)
            
            # Plot point
            axes[0].scatter(x, y, c=[colors1[i]], s=80, marker='o', 
                          edgecolors='white', linewidths=2, zorder=10)
            
            # Label
            label = f"{i+1}"
            axes[0].text(x, y, label, fontsize=10, fontweight='bold', 
                        ha='center', va='center', color='white', zorder=11)
            
            # Legend entry
            axes[0].scatter([], [], c=[colors1[i]], s=100, marker='o',
                          label=f"{i+1}: {vid} ({vclass})")
            
            frame_diff = abs(actual_frame - frame_num)
            print(f"  {i+1}. {vid:30s} ({vclass:15s}) at ({x:6.1f}, {y:6.1f}) [frame {actual_frame}, Δ={frame_diff}]")
        
        axes[0].legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        # Plot CSV2 on right
        axes[1].imshow(frame)
        axes[1].set_title(f'CSV2 Vehicles (Frame {frame_num})', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        print(f"\nCSV2 vehicles:")
        for i, vid in enumerate(vehicles2):
            x, y, actual_frame, vclass = self.get_vehicle_position_at_frame(self.df2, vid, frame_num)
            
            # Plot point
            axes[1].scatter(x, y, c=[colors2[i]], s=80, marker='s', 
                          edgecolors='white', linewidths=2, zorder=10)
            
            # Label
            label = f"{chr(65+i)}"  # A, B, C, ...
            axes[1].text(x, y, label, fontsize=10, fontweight='bold', 
                        ha='center', va='center', color='white', zorder=11)
            
            # Legend entry
            axes[1].scatter([], [], c=[colors2[i]], s=100, marker='s',
                          label=f"{chr(65+i)}: {vid} ({vclass})")
            
            frame_diff = abs(actual_frame - frame_num)
            print(f"  {chr(65+i)}. {vid:30s} ({vclass:15s}) at ({x:6.1f}, {y:6.1f}) [frame {actual_frame}, Δ={frame_diff}]")
        
        axes[1].legend(loc='upper left', fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved visualization: {output_path}")
        plt.close()
        
        # Also create a single overlay for comparison
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(frame)
        ax.set_title(f'Both CSVs Overlaid (Frame {frame_num})', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Plot CSV1 (circles)
        for i, vid in enumerate(vehicles1):
            x, y, _, _ = self.get_vehicle_position_at_frame(self.df1, vid, frame_num)
            ax.scatter(x, y, c=[colors1[i]], s=80, marker='o', 
                      edgecolors='white', linewidths=2, zorder=10, label=f"CSV1-{i+1}")
            ax.text(x, y, f"{i+1}", fontsize=9, fontweight='bold', 
                   ha='center', va='center', color='white', zorder=11)
        
        # Plot CSV2 (squares)
        for i, vid in enumerate(vehicles2):
            x, y, _, _ = self.get_vehicle_position_at_frame(self.df2, vid, frame_num)
            ax.scatter(x, y, c=[colors2[i]], s=80, marker='s', 
                      edgecolors='black', linewidths=2, zorder=10, label=f"CSV2-{chr(65+i)}")
            ax.text(x, y, f"{chr(65+i)}", fontsize=9, fontweight='bold', 
                   ha='center', va='center', color='black', zorder=11)
        
        # Add legend showing which is which
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='CSV1 (circles, 1-10)',
                   markerfacecolor='gray', markersize=10, markeredgecolor='white', markeredgewidth=2),
            Line2D([0], [0], marker='s', color='w', label='CSV2 (squares, A-J)',
                   markerfacecolor='gray', markersize=10, markeredgecolor='black', markeredgewidth=2)
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)
        
        overlay_path = output_path.parent / f"{output_path.stem}_overlay.png"
        plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved overlay: {overlay_path}")
        plt.close()
        
        print(f"\n" + "="*70)
        print("MANUAL MATCHING GUIDE:")
        print("="*70)
        print("Look at the overlay image and identify which vehicles are the same:")
        print("  • CSV1 vehicles are marked with circles (1-10)")
        print("  • CSV2 vehicles are marked with squares (A-J)")
        print("  • Find pairs that appear at the same location")
        print("  • Example: If circle '3' and square 'E' are on same vehicle, note: 3→E")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize sample trajectory points from two CSVs on video"
    )
    parser.add_argument("--csv1", required=True, help="First trajectory CSV")
    parser.add_argument("--csv2", required=True, help="Second trajectory CSV")
    parser.add_argument("--video", required=True, help="Video file")
    parser.add_argument("--output", default="output/trajectory_visualization.png", help="Output image path")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of vehicles to sample from each CSV")
    parser.add_argument("--frame", type=int, default=None, help="Specific frame number to use")
    
    args = parser.parse_args()
    
    viz = TrajectoryVisualizer(args.csv1, args.csv2, args.video)
    viz.visualize(n_samples=args.n_samples, output_path=args.output)


if __name__ == "__main__":
    main()
