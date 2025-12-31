#!/usr/bin/env python3
"""
Plot complete vehicle lifetimes - from appearance to disappearance
Compare generated trajectories with ground truth for 5 selected vehicles
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def calculate_trajectory_similarity(df1, df2, v1, v2):
    """Calculate similarity score between two vehicle trajectories"""
    data1 = df1[df1['VehicleID'] == v1].sort_values('Frame')
    data2 = df2[df2['VehicleID'] == v2].sort_values('Frame')
    
    frames1 = set(data1['Frame'])
    frames2 = set(data2['Frame'])
    common_frames = frames1.intersection(frames2)
    
    if len(common_frames) < 10:
        return None
    
    # Calculate average error in common frames
    total_error = 0
    count = 0
    for frame in common_frames:
        try:
            x1 = data1[data1['Frame'] == frame]['X_pixel'].iloc[0]
            y1 = data1[data1['Frame'] == frame]['Y_pixel'].iloc[0]
            x2 = data2[data2['Frame'] == frame]['X_pixel'].iloc[0]
            y2 = data2[data2['Frame'] == frame]['Y_pixel'].iloc[0]
            
            error = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            total_error += error
            count += 1
        except:
            continue
    
    if count == 0:
        return None
    
    avg_error = total_error / count
    overlap_ratio = len(common_frames) / max(len(frames1), len(frames2))
    
    return {
        'v1': v1,
        'v2': v2,
        'frames1': len(frames1),
        'frames2': len(frames2),
        'common': len(common_frames),
        'avg_error': avg_error,
        'overlap_ratio': overlap_ratio,
        'score': avg_error * (1 - overlap_ratio * 0.5)  # Lower is better
    }

def match_vehicles(df_generated, df_gt):
    """Match vehicles between generated and ground truth"""
    print("\n" + "="*70)
    print("MATCHING VEHICLES")
    print("="*70)
    
    vehicles_gen = df_generated['VehicleID'].unique()
    vehicles_gt = df_gt['VehicleID'].unique()
    
    print(f"Generated vehicles: {len(vehicles_gen)}")
    print(f"Ground truth vehicles: {len(vehicles_gt)}")
    
    matches = []
    
    for v_gen in vehicles_gen:
        for v_gt in vehicles_gt:
            result = calculate_trajectory_similarity(df_generated, df_gt, v_gen, v_gt)
            if result is not None and result['avg_error'] < 60 and result['common'] >= 30:
                matches.append(result)
    
    # Sort by score (lower is better)
    matches.sort(key=lambda x: x['score'])
    
    print(f"Found {len(matches)} good matches")
    
    return matches

def plot_vehicle_lifetime(df_generated, df_gt, v_gen, v_gt, output_path, vehicle_num):
    """Plot complete lifetime of a single vehicle"""
    data_gen = df_generated[df_generated['VehicleID'] == v_gen].sort_values('Frame')
    data_gt = df_gt[df_gt['VehicleID'] == v_gt].sort_values('Frame')
    
    # Get frame ranges
    gen_start = data_gen['Frame'].min()
    gen_end = data_gen['Frame'].max()
    gt_start = data_gt['Frame'].min()
    gt_end = data_gt['Frame'].max()
    
    # Get common frames
    common_frames = set(data_gen['Frame']).intersection(set(data_gt['Frame']))
    
    # Calculate errors in common frames
    errors = []
    for frame in sorted(common_frames):
        try:
            x1 = data_gen[data_gen['Frame'] == frame]['X_pixel'].iloc[0]
            y1 = data_gen[data_gen['Frame'] == frame]['Y_pixel'].iloc[0]
            x2 = data_gt[data_gt['Frame'] == frame]['X_pixel'].iloc[0]
            y2 = data_gt[data_gt['Frame'] == frame]['Y_pixel'].iloc[0]
            t = data_gt[data_gt['Frame'] == frame]['Time'].iloc[0]
            
            error = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            errors.append({'Frame': frame, 'Time': t, 'Error': error, 
                          'X1': x1, 'Y1': y1, 'X2': x2, 'Y2': y2})
        except:
            continue
    
    error_df = pd.DataFrame(errors)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(16, 14))
    
    avg_error = error_df['Error'].mean() if len(error_df) > 0 else 0
    overlap = len(common_frames)
    
    fig.suptitle(f'Vehicle {vehicle_num}: {v_gen} vs {v_gt}\n' + 
                 f'Generated: Frames {gen_start}-{gen_end} ({len(data_gen)} pts) | ' +
                 f'Ground Truth: Frames {gt_start}-{gt_end} ({len(data_gt)} pts)\n' +
                 f'Overlap: {overlap} frames | Avg Error: {avg_error:.1f}px',
                 fontsize=14, fontweight='bold')
    
    # ===== PLOT 1: SPATIAL TRAJECTORY (X vs Y) =====
    ax1 = axes[0]
    
    # Plot full trajectories
    ax1.plot(data_gen['X_pixel'], data_gen['Y_pixel'], 'b-', linewidth=3, alpha=0.7, 
             label=f'Generated (full: {len(data_gen)} frames)')
    ax1.plot(data_gt['X_pixel'], data_gt['Y_pixel'], 'r-', linewidth=3, alpha=0.7, 
             label=f'Ground Truth (full: {len(data_gt)} frames)')
    
    # Mark start and end points
    ax1.plot(data_gen['X_pixel'].iloc[0], data_gen['Y_pixel'].iloc[0], 'go', 
             markersize=15, label='Start (Gen)', zorder=5)
    ax1.plot(data_gen['X_pixel'].iloc[-1], data_gen['Y_pixel'].iloc[-1], 'mo', 
             markersize=15, label='End (Gen)', zorder=5)
    ax1.plot(data_gt['X_pixel'].iloc[0], data_gt['Y_pixel'].iloc[0], 'g^', 
             markersize=15, label='Start (GT)', zorder=5)
    ax1.plot(data_gt['X_pixel'].iloc[-1], data_gt['Y_pixel'].iloc[-1], 'm^', 
             markersize=15, label='End (GT)', zorder=5)
    
    # Highlight overlapping region
    if len(error_df) > 0:
        ax1.scatter(error_df['X1'], error_df['Y1'], c='cyan', s=30, alpha=0.5, 
                   label='Overlap region', zorder=4)
    
    ax1.set_xlabel('X Position (pixels)', fontsize=12)
    ax1.set_ylabel('Y Position (pixels)', fontsize=12)
    ax1.set_title('Complete Spatial Trajectory (Full Lifetime)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_aspect('equal', adjustable='box')
    
    # ===== PLOT 2: FRAME-BY-FRAME COMPARISON (X and Y positions) =====
    ax2 = axes[1]
    
    # X positions
    ax2_twin = ax2.twinx()
    
    # Plot X positions on left axis
    line1 = ax2.plot(data_gen['Frame'], data_gen['X_pixel'], 'b-', linewidth=2, 
                     alpha=0.7, label='Generated X')
    line2 = ax2.plot(data_gt['Frame'], data_gt['X_pixel'], 'r-', linewidth=2, 
                     alpha=0.7, label='Ground Truth X')
    
    # Plot Y positions on right axis
    line3 = ax2_twin.plot(data_gen['Frame'], data_gen['Y_pixel'], 'b--', linewidth=2, 
                          alpha=0.5, label='Generated Y')
    line4 = ax2_twin.plot(data_gt['Frame'], data_gt['Y_pixel'], 'r--', linewidth=2, 
                          alpha=0.5, label='Ground Truth Y')
    
    # Mark overlap region
    if len(common_frames) > 0:
        overlap_start = min(common_frames)
        overlap_end = max(common_frames)
        ax2.axvspan(overlap_start, overlap_end, alpha=0.1, color='green', 
                   label='Overlap region')
    
    ax2.set_xlabel('Frame Number', fontsize=12)
    ax2.set_ylabel('X Position (pixels)', fontsize=12, color='black')
    ax2_twin.set_ylabel('Y Position (pixels)', fontsize=12, color='gray')
    ax2.set_title('Position vs Frame Number (Full Lifetime)', fontsize=13, fontweight='bold')
    
    # Combine legends
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # ===== PLOT 3: ERROR ANALYSIS (only in overlap region) =====
    ax3 = axes[2]
    
    if len(error_df) > 0:
        # Euclidean error
        ax3.plot(error_df['Frame'], error_df['Error'], 'purple', linewidth=2.5, 
                label=f'Euclidean Error (Mean: {avg_error:.1f}px)')
        ax3.axhline(y=avg_error, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_error:.1f}px')
        
        # Error components
        x_error = np.abs(error_df['X1'] - error_df['X2'])
        y_error = np.abs(error_df['Y1'] - error_df['Y2'])
        
        ax3.fill_between(error_df['Frame'], 0, x_error, alpha=0.3, color='blue', 
                        label=f'X Error (Avg: {x_error.mean():.1f}px)')
        ax3.fill_between(error_df['Frame'], 0, y_error, alpha=0.3, color='orange', 
                        label=f'Y Error (Avg: {y_error.mean():.1f}px)')
        
        ax3.set_xlabel('Frame Number', fontsize=12)
        ax3.set_ylabel('Position Error (pixels)', fontsize=12)
        ax3.set_title(f'Tracking Error Over Time (Overlap: {len(common_frames)} frames)', 
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No overlapping frames', ha='center', va='center', 
                fontsize=16, transform=ax3.transAxes)
        ax3.set_title('No Overlap Between Trajectories', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Saved: {output_path.name}")
    
    return {
        'v_gen': v_gen,
        'v_gt': v_gt,
        'gen_frames': len(data_gen),
        'gt_frames': len(data_gt),
        'overlap': len(common_frames),
        'avg_error': avg_error
    }

def main():
    parser = argparse.ArgumentParser(description="Plot complete vehicle lifetimes")
    parser.add_argument('--generated', required=True, help="Generated trajectories CSV")
    parser.add_argument('--gt', required=True, help="Ground truth CSV")
    parser.add_argument('--output-dir', default=None, help="Output directory")
    parser.add_argument('--num-vehicles', type=int, default=5, help="Number of vehicles to plot")
    
    args = parser.parse_args()
    
    # Load CSVs
    print("\n" + "="*70)
    print("LOADING TRAJECTORIES")
    print("="*70)
    
    df_generated = pd.read_csv(args.generated)
    df_gt = pd.read_csv(args.gt)
    
    print(f"Generated: {len(df_generated)} points, {df_generated['VehicleID'].nunique()} vehicles")
    print(f"Ground Truth: {len(df_gt)} points, {df_gt['VehicleID'].nunique()} vehicles")
    
    # Filter ground truth to same frame range as generated
    min_frame = df_generated['Frame'].min()
    max_frame = df_generated['Frame'].max()
    df_gt = df_gt[(df_gt['Frame'] >= min_frame) & (df_gt['Frame'] <= max_frame)].copy()
    
    print(f"Ground Truth (filtered): {len(df_gt)} points, {df_gt['VehicleID'].nunique()} vehicles")
    
    # Match vehicles
    matches = match_vehicles(df_generated, df_gt)
    
    if len(matches) == 0:
        print("\n❌ No matches found!")
        return
    
    # Select vehicles to plot (best, worst, and some middle ones)
    num_vehicles = min(args.num_vehicles, len(matches))
    
    if num_vehicles <= 2:
        selected_matches = matches[:num_vehicles]
    else:
        # Select: best, worst, and evenly spaced middle ones
        best = matches[0]
        worst = matches[-1]
        middle_count = num_vehicles - 2
        step = max(1, len(matches) // (middle_count + 1))
        middle = [matches[i*step] for i in range(1, middle_count + 1)]
        selected_matches = [best] + middle + [worst]
    
    print(f"\n📊 Selected {num_vehicles} vehicles to plot:")
    for i, m in enumerate(selected_matches, 1):
        print(f"   {i}. {m['v1']} ↔ {m['v2']} | Overlap: {m['common']} | Error: {m['avg_error']:.1f}px")
    
    # Create output directory
    if args.output_dir is None:
        output_dir = Path(args.generated).parent / "vehicle_lifetimes"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot each vehicle
    print("\n" + "="*70)
    print("PLOTTING VEHICLE LIFETIMES")
    print("="*70)
    
    results = []
    
    for i, match in enumerate(selected_matches, 1):
        print(f"\n[{i}/{num_vehicles}] Plotting vehicle {match['v1']} vs {match['v2']}...")
        
        output_path = output_dir / f"vehicle_{i:02d}_{match['v1']}_vs_{match['v2']}.png"
        
        result = plot_vehicle_lifetime(
            df_generated, df_gt, 
            match['v1'], match['v2'], 
            output_path, i
        )
        
        results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📊 Generated {num_vehicles} plots")
    
    print(f"\n📈 Statistics:")
    for i, r in enumerate(results, 1):
        print(f"   Vehicle {i}: Gen={r['gen_frames']}f, GT={r['gt_frames']}f, "
              f"Overlap={r['overlap']}f ({r['overlap']/max(r['gen_frames'], r['gt_frames'])*100:.1f}%), "
              f"Error={r['avg_error']:.1f}px")
    
    avg_overlap = np.mean([r['overlap'] for r in results])
    avg_error = np.mean([r['avg_error'] for r in results if r['avg_error'] > 0])
    
    print(f"\n📊 Overall:")
    print(f"   Average overlap: {avg_overlap:.1f} frames")
    print(f"   Average error: {avg_error:.1f}px")
    
    print(f"\n✅ All plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
