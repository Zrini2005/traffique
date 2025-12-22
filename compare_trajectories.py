#!/usr/bin/env python3
"""
Compare Two Trajectory CSVs with Spatial Vehicle Matching

Matches vehicles between two tracking results based on spatial overlap,
then computes RMSE and other accuracy metrics.

Solves the problem: Vehicle IDs don't correspond between independent tracking runs.

Usage:
  python compare_trajectories.py --csv1 my_tracking.csv --csv2 friend_tracking.csv --output output/comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
from scipy.spatial.distance import cdist
import seaborn as sns

sns.set_style("whitegrid")


class TrajectoryComparator:
    """Compare two trajectory CSVs with automatic vehicle matching"""
    
    def __init__(self, csv1_path: str, csv2_path: str, format1: str = "auto", format2: str = "auto"):
        """
        Load two trajectory CSVs
        
        Args:
            csv1_path: First CSV (e.g., your tracking results)
            csv2_path: Second CSV (e.g., friend's tracking results)
            format1/format2: "default", "d2f1", or "auto" (auto-detect)
        """
        self.df1 = pd.read_csv(csv1_path)
        self.df2 = pd.read_csv(csv2_path)
        
        # Auto-detect format
        self.format1 = self._detect_format(self.df1) if format1 == "auto" else format1
        self.format2 = self._detect_format(self.df2) if format2 == "auto" else format2
        
        # Normalize to common column names
        self.df1 = self._normalize_columns(self.df1, self.format1, label="CSV1")
        self.df2 = self._normalize_columns(self.df2, self.format2, label="CSV2")
        
        print(f"✓ Loaded CSV1: {len(self.df1)} points ({self.format1} format)")
        print(f"✓ Loaded CSV2: {len(self.df2)} points ({self.format2} format)")
        print(f"✓ Frame range CSV1: {self.df1['frame'].min()} - {self.df1['frame'].max()}")
        print(f"✓ Frame range CSV2: {self.df2['frame'].min()} - {self.df2['frame'].max()}")
    
    def _detect_format(self, df: pd.DataFrame) -> str:
        """Auto-detect CSV format"""
        if 'X_pixel' in df.columns and 'Y_pixel' in df.columns:
            return "d2f1"
        elif 'x_px' in df.columns and 'y_px' in df.columns:
            return "default"
        else:
            raise ValueError(f"Unknown CSV format. Columns: {list(df.columns)}")
    
    def _normalize_columns(self, df: pd.DataFrame, format_type: str, label: str) -> pd.DataFrame:
        """Normalize column names to standard format"""
        df = df.copy()
        
        if format_type == "d2f1":
            df.rename(columns={
                'Frame': 'frame',
                'VehicleID': 'vehicle_id',
                'X_pixel': 'x_px',
                'Y_pixel': 'y_px',
                'Class': 'class'
            }, inplace=True)
        elif format_type == "default":
            # Already in standard format
            pass
        
        df['source'] = label
        return df
    
    def match_vehicles_spatially(self, iou_threshold: float = 0.3, min_overlap_frames: int = 5) -> Dict:
        """
        Match vehicles between CSV1 and CSV2 based on spatial trajectory overlap.
        
        Strategy:
        1. For each vehicle in CSV1, find all vehicles in CSV2 that appear in overlapping frames
        2. Compute trajectory IoU (Intersection over Union) for spatial overlap
        3. Match vehicles with highest IoU above threshold
        
        Returns:
            matches: {csv1_vehicle_id: csv2_vehicle_id}
        """
        
        print(f"\n🔗 Matching vehicles between CSVs...")
        print(f"   Parameters: IoU threshold={iou_threshold}, min overlap frames={min_overlap_frames}")
        
        # Get unique vehicles
        vehicles_1 = self.df1['vehicle_id'].unique()
        vehicles_2 = self.df2['vehicle_id'].unique()
        
        print(f"   CSV1 vehicles: {len(vehicles_1)}")
        print(f"   CSV2 vehicles: {len(vehicles_2)}")
        
        matches = {}
        match_scores = {}
        
        for v1 in vehicles_1:
            traj1 = self.df1[self.df1['vehicle_id'] == v1]
            frames1 = set(traj1['frame'].values)
            
            best_match = None
            best_score = 0.0
            
            for v2 in vehicles_2:
                traj2 = self.df2[self.df2['vehicle_id'] == v2]
                frames2 = set(traj2['frame'].values)
                
                # Check frame overlap
                common_frames = frames1.intersection(frames2)
                if len(common_frames) < min_overlap_frames:
                    continue
                
                # Compute spatial IoU in overlapping frames
                traj1_common = traj1[traj1['frame'].isin(common_frames)].sort_values('frame')
                traj2_common = traj2[traj2['frame'].isin(common_frames)].sort_values('frame')
                
                if len(traj1_common) == 0 or len(traj2_common) == 0:
                    continue
                
                # Compute average distance in overlapping frames
                distances = []
                for frame in common_frames:
                    p1 = traj1_common[traj1_common['frame'] == frame][['x_px', 'y_px']].values
                    p2 = traj2_common[traj2_common['frame'] == frame][['x_px', 'y_px']].values
                    
                    if len(p1) > 0 and len(p2) > 0:
                        dist = np.linalg.norm(p1[0] - p2[0])
                        distances.append(dist)
                
                if len(distances) == 0:
                    continue
                
                avg_distance = np.mean(distances)
                
                # Convert distance to IoU-like score (inverse distance, normalized)
                # Lower distance = higher score
                score = 1.0 / (1.0 + avg_distance / 100.0)  # Normalize by 100 pixels
                
                # Bonus for more overlapping frames
                score *= min(1.0, len(common_frames) / 50.0)
                
                if score > best_score and score > iou_threshold:
                    best_score = score
                    best_match = v2
            
            if best_match is not None:
                matches[v1] = best_match
                match_scores[v1] = best_score
        
        print(f"\n   ✅ Matched {len(matches)} vehicle pairs")
        if len(matches) > 0:
            print(f"   Average match score: {np.mean(list(match_scores.values())):.3f}")
            print(f"   Top 5 matches:")
            for v1, score in sorted(match_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      CSV1:{v1} → CSV2:{matches[v1]} (score={score:.3f})")
        
        return matches
    
    def compute_rmse_for_matches(self, matches: Dict) -> Dict:
        """Compute RMSE for each matched vehicle pair"""
        
        print(f"\n📐 Computing RMSE for matched pairs...")
        
        results = {}
        
        for v1, v2 in matches.items():
            traj1 = self.df1[self.df1['vehicle_id'] == v1].sort_values('frame')
            traj2 = self.df2[self.df2['vehicle_id'] == v2].sort_values('frame')
            
            # Find common frames
            frames1 = set(traj1['frame'].values)
            frames2 = set(traj2['frame'].values)
            common_frames = frames1.intersection(frames2)
            
            if len(common_frames) == 0:
                continue
            
            traj1_common = traj1[traj1['frame'].isin(common_frames)].sort_values('frame')
            traj2_common = traj2[traj2['frame'].isin(common_frames)].sort_values('frame')
            
            # Align by frame
            merged = pd.merge(
                traj1_common[['frame', 'x_px', 'y_px']],
                traj2_common[['frame', 'x_px', 'y_px']],
                on='frame',
                suffixes=('_1', '_2')
            )
            
            if len(merged) == 0:
                continue
            
            # Compute errors
            errors = np.sqrt(
                (merged['x_px_1'] - merged['x_px_2'])**2 +
                (merged['y_px_1'] - merged['y_px_2'])**2
            )
            
            results[f"{v1}->{v2}"] = {
                'v1': v1,
                'v2': v2,
                'n_points': len(merged),
                'rmse': np.sqrt(np.mean(errors**2)),
                'mae': np.mean(errors),
                'max_error': np.max(errors),
                'median_error': np.median(errors),
                'std_error': np.std(errors),
                'errors': errors.values
            }
        
        # Overall statistics
        if results:
            all_rmse = [r['rmse'] for r in results.values()]
            print(f"\n   Overall RMSE statistics:")
            print(f"      Mean RMSE: {np.mean(all_rmse):.2f} px")
            print(f"      Median RMSE: {np.median(all_rmse):.2f} px")
            print(f"      Std RMSE: {np.std(all_rmse):.2f} px")
            print(f"      Best (min): {np.min(all_rmse):.2f} px")
            print(f"      Worst (max): {np.max(all_rmse):.2f} px")
        
        return results
    
    def plot_comparison(self, matches: Dict, rmse_results: Dict, output_dir: str = "output/comparison"):
        """Generate comparison visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not rmse_results:
            print("No matched pairs to visualize")
            return
        
        # 1. RMSE distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        all_rmse = [r['rmse'] for r in rmse_results.values()]
        all_errors = np.concatenate([r['errors'] for r in rmse_results.values()])
        
        # RMSE per vehicle pair
        axes[0, 0].bar(range(len(all_rmse)), sorted(all_rmse, reverse=True))
        axes[0, 0].axhline(np.mean(all_rmse), color='red', linestyle='--', label=f'Mean: {np.mean(all_rmse):.1f}px')
        axes[0, 0].set_xlabel('Vehicle Pair (sorted)')
        axes[0, 0].set_ylabel('RMSE (pixels)')
        axes[0, 0].set_title(f'RMSE per Matched Vehicle Pair (n={len(all_rmse)})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[0, 1].hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(all_errors), color='red', linestyle='--', label=f'Mean: {np.mean(all_errors):.1f}px')
        axes[0, 1].axvline(np.median(all_errors), color='green', linestyle='--', label=f'Median: {np.median(all_errors):.1f}px')
        axes[0, 1].set_xlabel('Position Error (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Overall Error Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative error
        sorted_errors = np.sort(all_errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[1, 0].plot(sorted_errors, cumulative, linewidth=2)
        axes[1, 0].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].axhline(0.95, color='orange', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Position Error (pixels)')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # RMSE vs trajectory length
        lengths = [r['n_points'] for r in rmse_results.values()]
        rmses = [r['rmse'] for r in rmse_results.values()]
        axes[1, 1].scatter(lengths, rmses, alpha=0.6, s=100)
        axes[1, 1].set_xlabel('Trajectory Length (points)')
        axes[1, 1].set_ylabel('RMSE (pixels)')
        axes[1, 1].set_title('RMSE vs Trajectory Length')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'trajectory_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n  📊 Saved: trajectory_comparison.png")
        plt.close()
        
        # Save detailed results
        import json
        report = {
            'n_matched_pairs': len(matches),
            'overall_rmse_mean': float(np.mean(all_rmse)),
            'overall_rmse_median': float(np.median(all_rmse)),
            'overall_rmse_std': float(np.std(all_rmse)),
            'overall_mae': float(np.mean(all_errors)),
            'matched_pairs': {
                k: {key: float(val) if isinstance(val, (np.floating, np.integer)) else val 
                    for key, val in v.items() if key != 'errors'}
                for k, v in rmse_results.items()
            }
        }
        
        with open(output_path / 'comparison_metrics.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"  📄 Saved: comparison_metrics.json")
    
    def generate_report(self, output_dir: str = "output/comparison"):
        """Generate full comparison report"""
        
        print("\n" + "="*70)
        print("  TRAJECTORY COMPARISON ANALYSIS")
        print("="*70 + "\n")
        
        # Match vehicles
        matches = self.match_vehicles_spatially()
        
        if len(matches) == 0:
            print("\n❌ No vehicle matches found. Check that:")
            print("   • Both CSVs cover overlapping frame ranges")
            print("   • Vehicles appear in same spatial locations")
            print("   • Try lowering --iou-threshold")
            return
        
        # Compute RMSE
        rmse_results = self.compute_rmse_for_matches(matches)
        
        # Plot
        self.plot_comparison(matches, rmse_results, output_dir)
        
        print(f"\n✅ Comparison complete! Results saved to: {output_dir}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two trajectory CSVs with automatic vehicle matching"
    )
    parser.add_argument("--csv1", required=True, help="First trajectory CSV")
    parser.add_argument("--csv2", required=True, help="Second trajectory CSV (to compare against)")
    parser.add_argument("--output", default="output/comparison", help="Output directory")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="Matching threshold")
    parser.add_argument("--min-overlap", type=int, default=5, help="Minimum overlapping frames")
    
    args = parser.parse_args()
    
    comparator = TrajectoryComparator(args.csv1, args.csv2)
    comparator.generate_report(args.output)


if __name__ == "__main__":
    main()
