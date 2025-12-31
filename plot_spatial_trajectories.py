#!/usr/bin/env python3
"""
Generate spatial trajectory plots (X vs Y) from existing CSV files
Uses already generated trajectory CSV to avoid re-tracking
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def match_vehicles(df_generated, df_gt, min_overlap_frames=30, max_spatial_distance=80):
    """Match vehicles between generated and ground truth based on spatial-temporal overlap"""
    
    print("Matching vehicles...")
    
    gen_vehicles = df_generated['VehicleID'].unique()
    gt_vehicles = df_gt['VehicleID'].unique()
    
    matches = []
    
    for gen_id in gen_vehicles:
        gen_data = df_generated[df_generated['VehicleID'] == gen_id]
        gen_frames = set(gen_data['Frame'])
        gen_center = (gen_data['X_pixel'].mean(), gen_data['Y_pixel'].mean())
        
        best_match = None
        best_score = 0
        
        for gt_id in gt_vehicles:
            gt_data = df_gt[df_gt['VehicleID'] == gt_id]
            gt_frames = set(gt_data['Frame'])
            gt_center = (gt_data['X_pixel'].mean(), gt_data['Y_pixel'].mean())
            
            # Calculate temporal overlap
            overlap_frames = gen_frames.intersection(gt_frames)
            if len(overlap_frames) < min_overlap_frames:
                continue
            
            # Calculate spatial distance
            spatial_dist = np.sqrt((gen_center[0] - gt_center[0])**2 + 
                                  (gen_center[1] - gt_center[1])**2)
            
            if spatial_dist > max_spatial_distance:
                continue
            
            # Score based on overlap and proximity
            score = len(overlap_frames) / spatial_dist
            
            if score > best_score:
                best_score = score
                best_match = (gen_id, gt_id, len(overlap_frames), spatial_dist)
        
        if best_match:
            matches.append(best_match)
    
    print(f"Found {len(matches)} matches\n")
    return matches

def plot_spatial_trajectory(gen_data, gt_data, gen_id, gt_id, output_path):
    """Create X vs Y spatial trajectory plot"""
    
    # Get common frames
    common_frames = sorted(set(gen_data['Frame']).intersection(set(gt_data['Frame'])))
    
    gen_common = gen_data[gen_data['Frame'].isin(common_frames)].sort_values('Frame')
    gt_common = gt_data[gt_data['Frame'].isin(common_frames)].sort_values('Frame')
    
    # Calculate error
    euclidean_errors = np.sqrt((gen_common['X_pixel'].values - gt_common['X_pixel'].values)**2 +
                                (gen_common['Y_pixel'].values - gt_common['Y_pixel'].values)**2)
    mae = euclidean_errors.mean()
    rmse = np.sqrt((euclidean_errors**2).mean())
    
    # Create plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot full trajectories (including non-overlapping parts)
    ax.plot(gen_data['X_pixel'], gen_data['Y_pixel'], 'b-', linewidth=3, alpha=0.4, 
            label='Generated (full)')
    ax.plot(gt_data['X_pixel'], gt_data['Y_pixel'], 'r-', linewidth=3, alpha=0.4, 
            label='Ground Truth (full)')
    
    # Overlay common trajectory with markers
    ax.plot(gen_common['X_pixel'], gen_common['Y_pixel'], 'bo', markersize=5, alpha=0.7, 
            label='Generated (overlap)')
    ax.plot(gt_common['X_pixel'], gt_common['Y_pixel'], 'rs', markersize=5, alpha=0.7, 
            label='Ground Truth (overlap)')
    
    # Mark start and end points
    ax.plot(gen_common['X_pixel'].iloc[0], gen_common['Y_pixel'].iloc[0], 'go', 
            markersize=15, label='Start', zorder=5)
    ax.plot(gen_common['X_pixel'].iloc[-1], gen_common['Y_pixel'].iloc[-1], 'mo', 
            markersize=15, label='End', zorder=5)
    
    # Draw connection lines every N frames to show correspondence
    step = max(1, len(gen_common) // 10)  # Draw ~10 connection lines
    for i in range(0, len(gen_common), step):
        ax.plot([gen_common['X_pixel'].iloc[i], gt_common['X_pixel'].iloc[i]], 
                [gen_common['Y_pixel'].iloc[i], gt_common['Y_pixel'].iloc[i]], 
                'gray', linestyle='--', linewidth=1.5, alpha=0.4, zorder=1)
    
    ax.set_xlabel('X Position (pixels)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (pixels)', fontsize=14, fontweight='bold')
    ax.set_title(f'Spatial Trajectory: Gen {gen_id} vs GT {gt_id}\n'
                 f'MAE: {mae:.2f}px | RMSE: {rmse:.2f}px | Frames: {len(common_frames)}',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=1)
    ax.invert_yaxis()  # Invert Y axis (image coordinates)
    
    # Set appropriate axis limits with minimal padding - zoom into trajectory area
    all_x = pd.concat([gen_data['X_pixel'], gt_data['X_pixel']])
    all_y = pd.concat([gen_data['Y_pixel'], gt_data['Y_pixel']])
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    
    # Use smaller padding (5% instead of 10%) to keep trajectories closer
    ax.set_xlim(all_x.min() - 0.05*x_range, all_x.max() + 0.05*x_range)
    ax.set_ylim(all_y.max() + 0.05*y_range, all_y.min() - 0.05*y_range)  # Inverted
    
    # Add tick formatting for better readability
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    import sys
    
    # User can specify comparison directory, otherwise use comparison_001
    if len(sys.argv) > 1:
        comparison_dir = Path(sys.argv[1])
    else:
        comparison_dir = Path("output/comparison_001")
    
    if not comparison_dir.exists():
        print(f"❌ Directory not found: {comparison_dir}")
        print("\nAvailable directories:")
        for d in sorted(Path("output").glob("comparison_*")):
            print(f"   {d}")
        return
    
    generated_csv = comparison_dir / "generated_trajectories.csv"
    
    if not generated_csv.exists():
        print(f"❌ Generated CSV not found: {generated_csv}")
        return
    
    gt_csv_path = "/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv"
    
    print("="*70)
    print("SPATIAL TRAJECTORY PLOTS (X vs Y)")
    print("="*70)
    print(f"\nGenerated CSV: {generated_csv}")
    print(f"Ground Truth: {gt_csv_path}")
    print(f"Output Dir: {comparison_dir}\n")
    
    # Load CSVs
    print("Loading CSVs...")
    df_generated = pd.read_csv(generated_csv)
    df_gt = pd.read_csv(gt_csv_path)
    
    # Filter GT to same frame range
    min_frame = df_generated['Frame'].min()
    max_frame = df_generated['Frame'].max()
    df_gt = df_gt[(df_gt['Frame'] >= min_frame) & (df_gt['Frame'] <= max_frame)]
    
    print(f"Generated: {df_generated['VehicleID'].nunique()} vehicles")
    print(f"Ground Truth: {df_gt['VehicleID'].nunique()} vehicles\n")
    
    # Match vehicles
    matches = match_vehicles(df_generated, df_gt)
    
    if not matches:
        print("❌ No matches found!")
        return
    
    # Generate spatial plots for each match
    print("Generating spatial trajectory plots...")
    
    for i, (gen_id, gt_id, overlap, dist) in enumerate(matches, 1):
        gen_data = df_generated[df_generated['VehicleID'] == gen_id].sort_values('Frame')
        gt_data = df_gt[df_gt['VehicleID'] == gt_id].sort_values('Frame')
        
        plot_path = comparison_dir / f"spatial_{i}_gen{gen_id}_gt{gt_id}.png"
        plot_spatial_trajectory(gen_data, gt_data, gen_id, gt_id, plot_path)
        
        print(f"  [{i}/{len(matches)}] Gen:{gen_id} ↔ GT:{gt_id} (overlap: {overlap} frames)")
    
    print(f"\n✅ Generated {len(matches)} spatial trajectory plots")
    print(f"   Saved to: {comparison_dir}/spatial_*.png")
    print("="*70)

if __name__ == "__main__":
    main()
