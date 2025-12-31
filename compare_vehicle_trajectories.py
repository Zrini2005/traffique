import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist

def load_csv(file_path):
    """Load CSV and return dataframe."""
    df = pd.read_csv(file_path)
    print(f"Loaded {file_path}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"Vehicle IDs: {df['VehicleID'].nunique()}")
    return df

def match_vehicles(df1, df2, max_matches=10):
    """
    Match vehicles between two dataframes based on spatial and temporal overlap.
    Returns list of tuples: (vehicle_id_csv1, vehicle_id_csv2, similarity_score)
    OPTIMIZED: Pre-compute vehicle stats and use spatial binning.
    """
    print(f"\nPre-computing vehicle statistics...")
    
    # Pre-compute stats for all vehicles to avoid repeated calculations
    def get_vehicle_stats(df):
        stats = {}
        for vid in df['VehicleID'].unique():
            vdata = df[df['VehicleID'] == vid]
            stats[vid] = {
                'frames': set(vdata['Frame'].values),
                'frame_min': vdata['Frame'].min(),
                'frame_max': vdata['Frame'].max(),
                'x_mean': vdata['X_pixel'].mean(),
                'y_mean': vdata['Y_pixel'].mean(),
                'x_min': vdata['X_pixel'].min(),
                'x_max': vdata['X_pixel'].max(),
                'y_min': vdata['Y_pixel'].min(),
                'y_max': vdata['Y_pixel'].max(),
            }
        return stats
    
    stats1 = get_vehicle_stats(df1)
    stats2 = get_vehicle_stats(df2)
    
    print(f"CSV1: {len(stats1)} vehicles, CSV2: {len(stats2)} vehicles")
    print(f"Total comparisons needed: {len(stats1)} × {len(stats2)} = {len(stats1) * len(stats2):,}")
    print(f"Matching vehicles (with optimizations)...\n")
    
    matches = []
    checked = 0
    skipped_temporal = 0
    skipped_spatial = 0
    
    # Sort vehicles by frame range to enable early termination
    vehicles_csv1 = sorted(stats1.keys(), key=lambda v: stats1[v]['frame_min'])
    
    for i, vid1 in enumerate(vehicles_csv1):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(vehicles_csv1)} vehicles checked, {len(matches)} matches found")
        
        v1 = stats1[vid1]
        best_match = None
        best_score = float('inf')
        
        for vid2 in stats2.keys():
            v2 = stats2[vid2]
            
            # Quick temporal filter: check if frame ranges overlap
            if v1['frame_max'] < v2['frame_min'] or v1['frame_min'] > v2['frame_max']:
                skipped_temporal += 1
                continue
            
            # Quick spatial filter: check if bounding boxes are nearby (within 200 pixels)
            x_dist = max(0, v1['x_min'] - v2['x_max'], v2['x_min'] - v1['x_max'])
            y_dist = max(0, v1['y_min'] - v2['y_max'], v2['y_min'] - v1['y_max'])
            if x_dist > 200 or y_dist > 200:
                skipped_spatial += 1
                continue
            
            checked += 1
            
            # Check actual temporal overlap
            frame_overlap = len(v1['frames'].intersection(v2['frames']))
            if frame_overlap < 5:
                continue
            
            # Calculate spatial distance between centroids
            spatial_distance = np.sqrt((v1['x_mean'] - v2['x_mean'])**2 + 
                                      (v1['y_mean'] - v2['y_mean'])**2)
            
            # Combined score (lower is better)
            score = spatial_distance - (frame_overlap * 2)
            
            if score < best_score:
                best_score = score
                best_match = (vid1, vid2, frame_overlap, spatial_distance)
        
        if best_match:
            matches.append(best_match)
            # Early exit if we have enough good matches
            if len(matches) >= max_matches * 3:
                print(f"  Found {len(matches)} matches, stopping early search...")
                break
    
    print(f"\nPerformance stats:")
    print(f"  - Actual comparisons: {checked:,} (saved {skipped_temporal + skipped_spatial:,})")
    print(f"  - Skipped by temporal filter: {skipped_temporal:,}")
    print(f"  - Skipped by spatial filter: {skipped_spatial:,}")
    
    # Sort by overlap (descending) and spatial distance (ascending)
    matches.sort(key=lambda x: (-x[2], x[3]))
    
    print(f"\nTop {max_matches} matches:")
    for i, (vid1, vid2, overlap, dist) in enumerate(matches[:max_matches]):
        print(f"{i+1}. CSV1: {vid1} <-> CSV2: {vid2} (Overlap: {overlap} frames, Distance: {dist:.2f} pixels)")
    
    return matches[:max_matches]

def plot_vehicle_comparison(df1, df2, vehicle_id1, vehicle_id2, output_dir):
    """
    Create X vs Time and Y vs Time comparison plots for matched vehicles.
    """
    # Extract data for both vehicles
    v1_data = df1[df1['VehicleID'] == vehicle_id1].sort_values('Frame').copy()
    v2_data = df2[df2['VehicleID'] == vehicle_id2].sort_values('Frame').copy()
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'Vehicle Trajectory Comparison\nCSV1: {vehicle_id1} vs CSV2: {vehicle_id2}', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: X vs Time
    ax1.plot(v1_data['Time'], v1_data['X_pixel'], 'b-o', label=f'CSV1: {vehicle_id1}', 
             markersize=4, linewidth=2, alpha=0.7)
    ax1.plot(v2_data['Time'], v2_data['X_pixel'], 'r-s', label=f'CSV2: {vehicle_id2}', 
             markersize=4, linewidth=2, alpha=0.7)
    ax1.set_xlabel('Time (seconds)', fontsize=12)
    ax1.set_ylabel('X Position (pixels)', fontsize=12)
    ax1.set_title('X Position vs Time', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Y vs Time
    ax2.plot(v1_data['Time'], v1_data['Y_pixel'], 'b-o', label=f'CSV1: {vehicle_id1}', 
             markersize=4, linewidth=2, alpha=0.7)
    ax2.plot(v2_data['Time'], v2_data['Y_pixel'], 'r-s', label=f'CSV2: {vehicle_id2}', 
             markersize=4, linewidth=2, alpha=0.7)
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('Y Position (pixels)', fontsize=12)
    ax2.set_title('Y Position vs Time', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f'comparison_{vehicle_id1}_vs_{vehicle_id2}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    # File paths (WSL format)
    csv1_path = "/mnt/c/Users/srini/Downloads/trajectories_yash.csv"
    csv2_path = "/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv"
    output_dir = Path("/mnt/c/Users/srini/Downloads/trajectory_comparisons")
    
    # Create output directory
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {output_dir}")
    
    # Load CSVs
    print("\n" + "="*60)
    print("LOADING CSV FILES")
    print("="*60)
    df1 = load_csv(csv1_path)
    df2 = load_csv(csv2_path)
    
    # Match vehicles
    print("\n" + "="*60)
    print("MATCHING VEHICLES")
    print("="*60)
    matches = match_vehicles(df1, df2, max_matches=10)
    
    if not matches:
        print("\nNo vehicle matches found!")
        return
    
    # Generate plots for each match
    print("\n" + "="*60)
    print("GENERATING COMPARISON PLOTS")
    print("="*60)
    for i, (vid1, vid2, overlap, dist) in enumerate(matches, 1):
        print(f"\n[{i}/{len(matches)}] Plotting {vid1} vs {vid2}...")
        plot_vehicle_comparison(df1, df2, vid1, vid2, output_dir)
    
    print("\n" + "="*60)
    print(f"✓ COMPLETE! Generated {len(matches)} comparison plots")
    print(f"✓ Output saved to: {output_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
