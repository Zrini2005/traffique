"""
Detailed comparison between your tracking (CSV1) and ground truth (CSV2)
Generates frame-by-frame analysis and error metrics for matched vehicles
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_vehicle_pair(df1, df2, vid1, vid2, frame_limit=200):
    """
    Detailed analysis of one vehicle pair
    Returns comparison dataframe and metrics
    """
    # Get vehicle data
    v1 = df1[df1['VehicleID'] == vid1].sort_values('Frame').copy()
    v2 = df2[df2['VehicleID'] == vid2].sort_values('Frame').copy()
    
    # Limit to first N frames for analysis
    v1 = v1.head(frame_limit)
    v2 = v2.head(frame_limit)
    
    # Find common frames
    common_frames = sorted(set(v1['Frame']).intersection(set(v2['Frame'])))
    
    if len(common_frames) < 10:
        return None, None
    
    # Get data for common frames
    v1_common = v1[v1['Frame'].isin(common_frames)].set_index('Frame')
    v2_common = v2[v2['Frame'].isin(common_frames)].set_index('Frame')
    
    # Calculate errors
    errors = []
    for frame in common_frames:
        x1, y1 = v1_common.loc[frame, 'X_pixel'], v1_common.loc[frame, 'Y_pixel']
        x2, y2 = v2_common.loc[frame, 'X_pixel'], v2_common.loc[frame, 'Y_pixel']
        t = v2_common.loc[frame, 'Time']
        
        x_error = abs(x1 - x2)
        y_error = abs(y1 - y2)
        euclidean_error = np.sqrt(x_error**2 + y_error**2)
        
        errors.append({
            'Frame': frame,
            'Time': t,
            'X_CSV1': x1,
            'Y_CSV1': y1,
            'X_CSV2': x2,
            'Y_CSV2': y2,
            'X_Error': x_error,
            'Y_Error': y_error,
            'Euclidean_Error': euclidean_error
        })
    
    error_df = pd.DataFrame(errors)
    
    # Calculate metrics
    metrics = {
        'vehicle_csv1': vid1,
        'vehicle_csv2': vid2,
        'frames_analyzed': len(common_frames),
        'mean_x_error': error_df['X_Error'].mean(),
        'mean_y_error': error_df['Y_Error'].mean(),
        'mean_euclidean_error': error_df['Euclidean_Error'].mean(),
        'max_euclidean_error': error_df['Euclidean_Error'].max(),
        'median_euclidean_error': error_df['Euclidean_Error'].median(),
        'std_euclidean_error': error_df['Euclidean_Error'].std()
    }
    
    return error_df, metrics

def plot_detailed_comparison(error_df, metrics, output_path):
    """Create detailed comparison plots"""
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle(f"Detailed Trajectory Comparison: {metrics['vehicle_csv1']} vs {metrics['vehicle_csv2']}\n" + 
                 f"Mean Error: {metrics['mean_euclidean_error']:.1f}px | Max Error: {metrics['max_euclidean_error']:.1f}px",
                 fontsize=14, fontweight='bold')
    
    # Plot 1: X Position Comparison
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(error_df['Time'], error_df['X_CSV1'], 'b-o', label='CSV1 (Your Tracking)', 
             markersize=3, linewidth=1.5, alpha=0.7)
    ax1.plot(error_df['Time'], error_df['X_CSV2'], 'r-s', label='CSV2 (Ground Truth)', 
             markersize=3, linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('X Position (pixels)')
    ax1.set_title('X Position Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Y Position Comparison
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(error_df['Time'], error_df['Y_CSV1'], 'b-o', label='CSV1 (Your Tracking)', 
             markersize=3, linewidth=1.5, alpha=0.7)
    ax2.plot(error_df['Time'], error_df['Y_CSV2'], 'r-s', label='CSV2 (Ground Truth)', 
             markersize=3, linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Y Position (pixels)')
    ax2.set_title('Y Position Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: X Error over time
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(error_df['Time'], error_df['X_Error'], 'purple', linewidth=2)
    ax3.axhline(y=error_df['X_Error'].mean(), color='red', linestyle='--', 
                label=f"Mean: {error_df['X_Error'].mean():.1f}px")
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('X Error (pixels)')
    ax3.set_title('X Position Error Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Y Error over time
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(error_df['Time'], error_df['Y_Error'], 'orange', linewidth=2)
    ax4.axhline(y=error_df['Y_Error'].mean(), color='red', linestyle='--', 
                label=f"Mean: {error_df['Y_Error'].mean():.1f}px")
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Y Error (pixels)')
    ax4.set_title('Y Position Error Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Euclidean Error over time
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(error_df['Time'], error_df['Euclidean_Error'], 'darkred', linewidth=2)
    ax5.axhline(y=error_df['Euclidean_Error'].mean(), color='blue', linestyle='--', 
                label=f"Mean: {error_df['Euclidean_Error'].mean():.1f}px")
    ax5.axhline(y=error_df['Euclidean_Error'].median(), color='green', linestyle=':', 
                label=f"Median: {error_df['Euclidean_Error'].median():.1f}px")
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Euclidean Error (pixels)')
    ax5.set_title('Total Position Error Over Time')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Error distribution
    ax6 = plt.subplot(3, 2, 6)
    ax6.hist(error_df['Euclidean_Error'], bins=30, color='darkblue', alpha=0.7, edgecolor='black')
    ax6.axvline(x=error_df['Euclidean_Error'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f"Mean: {error_df['Euclidean_Error'].mean():.1f}px")
    ax6.axvline(x=error_df['Euclidean_Error'].median(), color='green', linestyle=':', 
                linewidth=2, label=f"Median: {error_df['Euclidean_Error'].median():.1f}px")
    ax6.set_xlabel('Euclidean Error (pixels)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    # Load CSVs
    csv1_path = "/mnt/c/Users/srini/Downloads/trajectories_yash.csv"
    csv2_path = "/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv"
    output_dir = Path("/mnt/c/Users/srini/Downloads/trajectory_analysis")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("DETAILED TRAJECTORY COMPARISON ANALYSIS")
    print("="*70)
    
    df1 = pd.read_csv(csv1_path)
    df2 = pd.read_csv(csv2_path)
    
    print(f"\nCSV1: {len(df1)} rows, {df1['VehicleID'].nunique()} vehicles")
    print(f"CSV2: {len(df2)} rows, {df2['VehicleID'].nunique()} vehicles")
    
    # Get matches from previous analysis (hardcoded top 5)
    matches = [
        ('car_19', 'Truck_3'),
        ('car_14', 'Truck_4'),
        ('car_10', 'Car_25'),
        ('car_17', 'Car_9'),
        ('car_12', 'Car_26')
    ]
    
    all_metrics = []
    
    for i, (vid1, vid2) in enumerate(matches, 1):
        print(f"\n{'='*70}")
        print(f"ANALYZING PAIR {i}/5: {vid1} vs {vid2}")
        print(f"{'='*70}")
        
        error_df, metrics = analyze_vehicle_pair(df1, df2, vid1, vid2, frame_limit=200)
        
        if error_df is None:
            print(f"⚠️  Not enough common frames for analysis")
            continue
        
        # Print metrics
        print(f"\n📊 ERROR METRICS:")
        print(f"   Frames analyzed: {metrics['frames_analyzed']}")
        print(f"   Mean X Error: {metrics['mean_x_error']:.2f} pixels")
        print(f"   Mean Y Error: {metrics['mean_y_error']:.2f} pixels")
        print(f"   Mean Euclidean Error: {metrics['mean_euclidean_error']:.2f} pixels")
        print(f"   Median Euclidean Error: {metrics['median_euclidean_error']:.2f} pixels")
        print(f"   Max Euclidean Error: {metrics['max_euclidean_error']:.2f} pixels")
        print(f"   Std Dev: {metrics['std_euclidean_error']:.2f} pixels")
        
        # Save detailed plots
        plot_path = output_dir / f"detailed_analysis_{vid1}_vs_{vid2}.png"
        plot_detailed_comparison(error_df, metrics, plot_path)
        print(f"\n✅ Saved detailed plot: {plot_path.name}")
        
        # Save error data
        csv_path = output_dir / f"errors_{vid1}_vs_{vid2}.csv"
        error_df.to_csv(csv_path, index=False)
        print(f"✅ Saved error data: {csv_path.name}")
        
        all_metrics.append(metrics)
    
    # Summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)
    
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics)
        print(f"\nOverall Statistics:")
        print(f"   Average Mean Error: {summary_df['mean_euclidean_error'].mean():.2f} pixels")
        print(f"   Average Max Error: {summary_df['max_euclidean_error'].mean():.2f} pixels")
        print(f"   Worst Case Error: {summary_df['max_euclidean_error'].max():.2f} pixels")
        
        # Save summary
        summary_path = output_dir / "summary_metrics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Saved summary: {summary_path}")
        
        # Diagnosis
        print("\n" + "="*70)
        print("🔍 DIAGNOSIS")
        print("="*70)
        avg_error = summary_df['mean_euclidean_error'].mean()
        
        if avg_error > 500:
            print("❌ CRITICAL: Your tracking is EXTREMELY inaccurate (>500px error)")
            print("   Possible issues:")
            print("   - Wrong homography/calibration")
            print("   - ID switches (tracking different vehicles)")
            print("   - Detection failures")
            print("   - Wrong camera coordinates")
        elif avg_error > 100:
            print("⚠️  WARNING: High tracking error (>100px)")
            print("   Possible issues:")
            print("   - Noisy detections")
            print("   - Poor smoothing")
            print("   - Calibration issues")
        elif avg_error > 50:
            print("⚠️  MODERATE: Tracking could be improved (>50px)")
        else:
            print("✅ GOOD: Tracking is relatively accurate (<50px)")
        
        print(f"\n📁 All results saved to: {output_dir}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
