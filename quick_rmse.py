#!/usr/bin/env python3
"""
Fast RMSE calculator for trajectory comparison
Uses simple spatial matching without complex iteration
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def calculate_rmse(csv1_path, csv2_path, output_dir="output/rmse_results"):
    """Calculate RMSE between two trajectory CSVs"""
    
    print("\n" + "="*70)
    print("  FAST TRAJECTORY RMSE CALCULATOR")
    print("="*70 + "\n")
    
    # Load CSVs
    print(f"Loading CSV1: {csv1_path}")
    df1 = pd.read_csv(csv1_path)
    print(f"  Points: {len(df1)}, Vehicles: {df1['VehicleID'].nunique()}")
    
    print(f"\nLoading CSV2: {csv2_path}")
    df2 = pd.read_csv(csv2_path)
    print(f"  Points: {len(df2)}, Vehicles: {df2['VehicleID'].nunique()}")
    
    # Merge on frame to find overlapping detections
    print("\n🔗 Finding spatial overlaps...")
    merged = pd.merge(
        df1[['Frame', 'VehicleID', 'X_world', 'Y_world']],
        df2[['Frame', 'VehicleID', 'X_world', 'Y_world']],
        on='Frame',
        suffixes=('_1', '_2')
    )
    
    print(f"  Overlapping frame detections: {len(merged)}")
    
    # Calculate distances for each pair
    merged['distance'] = np.sqrt(
        (merged['X_world_1'] - merged['X_world_2'])**2 +
        (merged['Y_world_1'] - merged['Y_world_2'])**2
    )
    
    # Find close matches (within 5 meters in same frame = likely same vehicle)
    threshold = 5.0  # meters
    matches = merged[merged['distance'] < threshold].copy()
    
    print(f"  Close matches (< {threshold}m): {len(matches)}")
    
    if len(matches) == 0:
        print("\n❌ No matching vehicles found!")
        return
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean(matches['distance']**2))
    mean_error = matches['distance'].mean()
    median_error = matches['distance'].median()
    max_error = matches['distance'].max()
    
    print(f"\n📊 RESULTS:")
    print("="*70)
    print(f"  Matched points: {len(matches)}")
    print(f"  RMSE: {rmse:.4f} meters")
    print(f"  Mean error: {mean_error:.4f} meters")
    print(f"  Median error: {median_error:.4f} meters")
    print(f"  Max error: {max_error:.4f} meters")
    print(f"  Min error: {matches['distance'].min():.4f} meters")
    print("="*70)
    
    # Error distribution
    print(f"\n📈 Error Distribution:")
    print(f"  < 0.5m: {(matches['distance'] < 0.5).sum()} ({(matches['distance'] < 0.5).sum()/len(matches)*100:.1f}%)")
    print(f"  < 1.0m: {(matches['distance'] < 1.0).sum()} ({(matches['distance'] < 1.0).sum()/len(matches)*100:.1f}%)")
    print(f"  < 2.0m: {(matches['distance'] < 2.0).sum()} ({(matches['distance'] < 2.0).sum()/len(matches)*100:.1f}%)")
    print(f"  < 5.0m: {(matches['distance'] < 5.0).sum()} ({(matches['distance'] < 5.0).sum()/len(matches)*100:.1f}%)")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'rmse_meters': float(rmse),
        'mean_error_meters': float(mean_error),
        'median_error_meters': float(median_error),
        'max_error_meters': float(max_error),
        'min_error_meters': float(matches['distance'].min()),
        'matched_points': int(len(matches)),
        'total_csv1_points': int(len(df1)),
        'total_csv2_points': int(len(df2))
    }
    
    import json
    with open(output_path / 'rmse_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Saved results to: {output_path / 'rmse_results.json'}")
    print("\n" + "="*70)
    
    return rmse

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python quick_rmse.py <csv1> <csv2> [output_dir]")
        sys.exit(1)
    
    csv1 = sys.argv[1]
    csv2 = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "output/rmse_results"
    
    calculate_rmse(csv1, csv2, output_dir)
