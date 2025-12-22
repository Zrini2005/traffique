#!/usr/bin/env python3
"""
Test different homography approaches to match D2F1_lclF.csv world coordinates

This script helps determine the correct homography transformation by:
1. Testing camera_calibration.json matrices
2. Computing a new homography from ground control points

Goal: Match X_world (1052-1255m) and Y_world (0-8.6m) ranges from D2F1_lclF.csv
"""

import numpy as np
import cv2
import pandas as pd
import json
from pathlib import Path


def test_homography_on_sample_points(H: np.ndarray, sample_pixels: list) -> dict:
    """
    Test homography transformation on sample pixel coordinates
    
    Args:
        H: 3x3 homography matrix
        sample_pixels: List of [x, y] pixel coordinates
    
    Returns:
        dict with transformed world coordinates
    """
    
    results = {
        'pixel_coords': [],
        'world_coords': [],
        'x_world': [],
        'y_world': []
    }
    
    for px, py in sample_pixels:
        # Homogeneous coordinates
        point_h = np.array([px, py, 1.0])
        world_h = H @ point_h
        
        # Normalize
        wx = float(world_h[0] / world_h[2])
        wy = float(world_h[1] / world_h[2])
        
        results['pixel_coords'].append([px, py])
        results['world_coords'].append([wx, wy])
        results['x_world'].append(wx)
        results['y_world'].append(wy)
    
    return results


def analyze_d2f1_coordinate_system():
    """Analyze D2F1_lclF.csv to understand the coordinate system"""
    
    print("\n" + "="*70)
    print("  ANALYZING D2F1_lclF.csv COORDINATE SYSTEM")
    print("="*70 + "\n")
    
    df = pd.read_csv('output/D2F1_lclF.csv')
    
    print("📊 Dataset Summary:")
    print(f"   Frames: {df['Frame'].min()} → {df['Frame'].max()} ({len(df['Frame'].unique())} unique)")
    print(f"   Vehicles: {df['VehicleID'].nunique()}")
    print(f"   Total points: {len(df)}")
    
    print("\n📐 Pixel Coordinates:")
    print(f"   X_pixel: {df['X_pixel'].min():.1f} → {df['X_pixel'].max():.1f} (range: {df['X_pixel'].max() - df['X_pixel'].min():.1f})")
    print(f"   Y_pixel: {df['Y_pixel'].min():.1f} → {df['Y_pixel'].max():.1f} (range: {df['Y_pixel'].max() - df['Y_pixel'].min():.1f})")
    
    print("\n🌍 World Coordinates:")
    print(f"   X_world: {df['X_world'].min():.2f} → {df['X_world'].max():.2f} meters (range: {df['X_world'].max() - df['X_world'].min():.2f}m)")
    print(f"   Y_world: {df['Y_world'].min():.2f} → {df['Y_world'].max():.2f} meters (range: {df['Y_world'].max() - df['Y_world'].min():.2f}m)")
    
    # Sample points from corners and center
    print("\n📍 Sample Points (Pixel → World):")
    
    # Find points at different pixel locations
    samples = [
        ('Left edge', df.nsmallest(1, 'X_pixel')),
        ('Right edge', df.nlargest(1, 'X_pixel')),
        ('Top edge', df.nsmallest(1, 'Y_pixel')),
        ('Bottom edge', df.nlargest(1, 'Y_pixel')),
        ('Center', df.iloc[len(df)//2:len(df)//2+1])
    ]
    
    for label, sample_df in samples:
        if not sample_df.empty:
            row = sample_df.iloc[0]
            px, py = row['X_pixel'], row['Y_pixel']
            wx, wy = row['X_world'], row['Y_world']
            
            # Compute apparent scale at this point
            print(f"   {label:12s}: ({px:7.1f}, {py:7.1f})px → ({wx:8.2f}, {wy:6.2f})m")
    
    # Compute scale variation
    print("\n📏 Scale Analysis (meters per pixel):")
    
    # Approximate scale at different locations
    left_df = df[df['X_pixel'] < 500]
    right_df = df[df['X_pixel'] > 3500]
    
    if not left_df.empty and not right_df.empty:
        left_scale_x = (left_df['X_world'].max() - left_df['X_world'].min()) / (left_df['X_pixel'].max() - left_df['X_pixel'].min())
        right_scale_x = (right_df['X_world'].max() - right_df['X_world'].min()) / (right_df['X_pixel'].max() - right_df['X_pixel'].min())
        
        print(f"   X scale (left):  {left_scale_x:.4f} m/px")
        print(f"   X scale (right): {right_scale_x:.4f} m/px")
        print(f"   Variation: {abs(left_scale_x - right_scale_x) / ((left_scale_x + right_scale_x)/2) * 100:.1f}%")
    
    return {
        'x_world_range': (df['X_world'].min(), df['X_world'].max()),
        'y_world_range': (df['Y_world'].min(), df['Y_world'].max()),
        'x_pixel_range': (df['X_pixel'].min(), df['X_pixel'].max()),
        'y_pixel_range': (df['Y_pixel'].min(), df['Y_pixel'].max())
    }


def create_reference_homography():
    """
    Create a reference homography based on D2F1_lclF.csv corner points
    
    This uses actual data points to reverse-engineer the homography
    """
    
    print("\n" + "="*70)
    print("  REVERSE-ENGINEERING HOMOGRAPHY FROM D2F1_lclF.csv")
    print("="*70 + "\n")
    
    df = pd.read_csv('output/D2F1_lclF.csv')
    
    # Select well-distributed points across the frame
    # Strategy: pick points from different spatial regions
    
    print("🔍 Selecting ground control points from D2F1_lclF.csv...")
    
    # Divide frame into grid and take one point from each region
    x_bins = np.linspace(df['X_pixel'].min(), df['X_pixel'].max(), 4)
    y_bins = np.linspace(df['Y_pixel'].min(), df['Y_pixel'].max(), 3)
    
    src_points = []
    dst_points = []
    
    for i in range(len(x_bins)-1):
        for j in range(len(y_bins)-1):
            # Find points in this bin
            mask = (
                (df['X_pixel'] >= x_bins[i]) & (df['X_pixel'] < x_bins[i+1]) &
                (df['Y_pixel'] >= y_bins[j]) & (df['Y_pixel'] < y_bins[j+1])
            )
            
            if mask.sum() > 0:
                # Take median point in this region
                subset = df[mask]
                median_idx = subset['X_pixel'].argsort().iloc[len(subset)//2]
                point = subset.iloc[median_idx]
                
                src_points.append([point['X_pixel'], point['Y_pixel']])
                dst_points.append([point['X_world'], point['Y_world']])
                
                print(f"   Point {len(src_points)}: ({point['X_pixel']:.1f}, {point['Y_pixel']:.1f})px "
                      f"→ ({point['X_world']:.2f}, {point['Y_world']:.2f})m")
    
    if len(src_points) < 4:
        print("❌ Not enough points found (need at least 4)")
        return None
    
    # Compute homography
    src_pts = np.array(src_points, dtype=np.float32)
    dst_pts = np.array(dst_points, dtype=np.float32)
    
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    print(f"\n✅ Computed homography from {len(src_points)} points")
    
    # Compute reprojection error
    errors = []
    for src, dst in zip(src_pts, dst_pts):
        src_h = np.array([src[0], src[1], 1.0])
        dst_pred_h = H @ src_h
        dst_pred = dst_pred_h[:2] / dst_pred_h[2]
        error = np.linalg.norm(dst_pred - dst)
        errors.append(error)
    
    print(f"   Reprojection error: {np.mean(errors):.4f} ± {np.std(errors):.4f} meters")
    print(f"   Max error: {np.max(errors):.4f} meters")
    
    # Save to file
    output_data = {
        "homography_matrix": H.tolist(),
        "source": "image_pixels",
        "target": "world_meters (D2F1 coordinate system)",
        "points_used": len(src_points),
        "reprojection_error_mean": float(np.mean(errors)),
        "reprojection_error_std": float(np.std(errors)),
        "method": "reverse-engineered from D2F1_lclF.csv",
        "correspondences": [
            {
                "image": [float(s[0]), float(s[1])],
                "world": [float(d[0]), float(d[1])]
            }
            for s, d in zip(src_points, dst_points)
        ]
    }
    
    output_file = Path("output/homography_from_d2f1.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n💾 Saved homography to: {output_file}")
    print(f"\n📝 Usage:")
    print(f'   python trajectory_tracker.py <video> --homography "{output_file}" ...')
    
    return H


def main():
    print("\n" + "="*70)
    print("  HOMOGRAPHY TESTING & CALIBRATION TOOL")
    print("="*70 + "\n")
    
    # Step 1: Analyze D2F1 coordinate system
    d2f1_info = analyze_d2f1_coordinate_system()
    
    # Step 2: Reverse-engineer homography from D2F1 data
    H = create_reference_homography()
    
    if H is not None:
        print("\n" + "="*70)
        print("  ✅ SUCCESS!")
        print("="*70)
        print("\nYou can now use the homography:")
        print("  output/homography_from_d2f1.json")
        print("\nNext steps:")
        print("  1. Run tracking with homography:")
        print('     python trajectory_tracker.py <video> --homography output/homography_from_d2f1.json --format d2f1 ...')
        print("  2. Compare results with D2F1_lclF.csv:")
        print('     python compare_trajectories.py --csv1 <your_output> --csv2 output/D2F1_lclF.csv ...')


if __name__ == "__main__":
    main()
