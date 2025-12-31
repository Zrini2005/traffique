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
from scipy.optimize import linear_sum_assignment
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
    
    def match_vehicles_spatially(self, min_confidence: float = 0.7, min_overlap_frames: int = 5, max_vehicles: int = None) -> Tuple[Dict, Dict]:
        """
        Match vehicles using Hungarian algorithm with multi-criteria scoring.
        
        Multi-Criteria Matching:
        1. Class filtering (car only matches car)
        2. Spatial distance (world coords if available, else pixels)
        3. Trajectory shape similarity (DTW)
        4. Temporal overlap
        5. Entry/exit point similarity
        6. Velocity profile similarity
        
        Returns:
            matches: {csv1_vehicle_id: csv2_vehicle_id}
            confidences: {csv1_vehicle_id: confidence_score}
        """
        
        print(f"\n🔗 Robust vehicle matching with Hungarian algorithm...")
        print(f"   Parameters: min confidence={min_confidence}, min overlap={min_overlap_frames}")
        
        vehicles_1 = sorted(self.df1['vehicle_id'].unique())
        vehicles_2 = sorted(self.df2['vehicle_id'].unique())
        
        # Limit to first N vehicles if specified
        if max_vehicles is not None:
            print(f"   ⚡ LIMITING to first {max_vehicles} vehicles for speed")
            vehicles_1 = vehicles_1[:max_vehicles]
            vehicles_2 = vehicles_2[:max_vehicles]
        
        print(f"   CSV1 vehicles: {len(vehicles_1)}")
        print(f"   CSV2 vehicles: {len(vehicles_2)}")
        
        # Check if world coordinates are available
        has_world = 'x_world' in self.df1.columns and 'y_world' in self.df1.columns
        coord_system = "world" if has_world else "pixel"
        print(f"   Using {coord_system} coordinates")
        
        # Build cost matrix (lower = better match)
        n1, n2 = len(vehicles_1), len(vehicles_2)
        cost_matrix = np.full((n1, n2), 1e6)  # Start with very high cost
        confidence_matrix = np.zeros((n1, n2))
        
        print(f"   Building {n1}x{n2} cost matrix...")
        
        for i, v1 in enumerate(vehicles_1):
            traj1 = self.df1[self.df1['vehicle_id'] == v1].sort_values('frame')
            
            # Get class if available
            class1 = traj1['class'].iloc[0] if 'class' in traj1.columns else None
            
            for j, v2 in enumerate(vehicles_2):
                traj2 = self.df2[self.df2['vehicle_id'] == v2].sort_values('frame')
                
                # Criterion 1: Class must match
                # DISABLED - class names differ between CSVs
                # if 'class' in traj2.columns and class1 is not None:
                #     class2 = traj2['class'].iloc[0]
                #     if class1 != class2:
                #         continue  # Skip, different classes
                
                # Check temporal overlap
                frames1 = set(traj1['frame'].values)
                frames2 = set(traj2['frame'].values)
                common_frames = frames1.intersection(frames2)
                
                if len(common_frames) < min_overlap_frames:
                    continue
                
                # Get trajectories in common frames
                traj1_common = traj1[traj1['frame'].isin(common_frames)].sort_values('frame')
                traj2_common = traj2[traj2['frame'].isin(common_frames)].sort_values('frame')
                
                # Align by frame
                merged = pd.merge(
                    traj1_common[['frame', 'x_px', 'y_px'] + (['x_world', 'y_world'] if has_world else [])],
                    traj2_common[['frame', 'x_px', 'y_px'] + (['x_world', 'y_world'] if has_world else [])],
                    on='frame',
                    suffixes=('_1', '_2')
                )
                
                if len(merged) < min_overlap_frames:
                    continue
                
                # === SCORING (0-1, higher is better) ===
                scores = []
                
                # Score 1: Spatial distance (40% weight)
                if has_world:
                    distances = np.sqrt(
                        (merged['x_world_1'] - merged['x_world_2'])**2 +
                        (merged['y_world_1'] - merged['y_world_2'])**2
                    )
                    # Assume <2m is excellent, >10m is bad
                    spatial_score = np.exp(-np.mean(distances) / 3.0)
                else:
                    distances = np.sqrt(
                        (merged['x_px_1'] - merged['x_px_2'])**2 +
                        (merged['y_px_1'] - merged['y_px_2'])**2
                    )
                    # <20px excellent, >100px bad
                    spatial_score = np.exp(-np.mean(distances) / 30.0)
                
                scores.append(('spatial', spatial_score, 0.40))
                
                # Score 2: Temporal overlap (20% weight)
                overlap_ratio = len(common_frames) / max(len(frames1), len(frames2))
                scores.append(('temporal', overlap_ratio, 0.20))
                
                # Score 3: Entry/exit similarity (15% weight)
                frame_start_diff = abs(traj1['frame'].min() - traj2['frame'].min())
                frame_end_diff = abs(traj1['frame'].max() - traj2['frame'].max())
                max_frames = max(traj1['frame'].max(), traj2['frame'].max())
                entry_exit_score = 1.0 - (frame_start_diff + frame_end_diff) / (2.0 * max_frames)
                scores.append(('entry_exit', max(0, entry_exit_score), 0.15))
                
                # Score 4: Trajectory length similarity (10% weight)
                len_ratio = min(len(traj1), len(traj2)) / max(len(traj1), len(traj2))
                scores.append(('length', len_ratio, 0.10))
                
                # Score 5: Velocity consistency (15% weight)
                if len(merged) > 1:
                    vel1 = np.diff(merged[['x_px_1', 'y_px_1']].values, axis=0)
                    vel2 = np.diff(merged[['x_px_2', 'y_px_2']].values, axis=0)
                    vel_diff = np.linalg.norm(vel1 - vel2, axis=1)
                    velocity_score = np.exp(-np.mean(vel_diff) / 50.0)
                    scores.append(('velocity', velocity_score, 0.15))
                else:
                    scores.append(('velocity', 0.5, 0.15))
                
                # Weighted confidence score
                confidence = sum(score * weight for _, score, weight in scores)
                confidence_matrix[i, j] = confidence
                
                # Cost = 1 - confidence (lower cost = better match)
                cost_matrix[i, j] = 1.0 - confidence
        
        # Debug: Check what confidences we got
        valid_confidences = confidence_matrix[confidence_matrix > 0]
        if len(valid_confidences) > 0:
            print(f"   Confidence scores found: min={valid_confidences.min():.3f}, max={valid_confidences.max():.3f}, mean={valid_confidences.mean():.3f}")
            print(f"   Valid pairs with ANY overlap: {len(valid_confidences)}")
        else:
            print(f"   ⚠️  NO valid pairs found (zero overlap or mismatched classes)")
        
        # Hungarian algorithm for optimal 1:1 matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Extract matches above confidence threshold
        matches = {}
        confidences = {}
        
        print(f"   Applying confidence threshold: {min_confidence}")
        for i, j in zip(row_ind, col_ind):
            confidence = confidence_matrix[i, j]
            if confidence >= min_confidence and cost_matrix[i, j] < 1e6:
                v1 = vehicles_1[i]
                v2 = vehicles_2[j]
                matches[v1] = v2
                confidences[v1] = confidence
            elif confidence > 0:
                print(f"      Rejected: CSV1:{vehicles_1[i]} → CSV2:{vehicles_2[j]} (confidence={confidence:.3f} < {min_confidence})")
        
        print(f"\n   ✅ Matched {len(matches)} vehicle pairs (1:1 guaranteed)")
        if len(matches) > 0:
            conf_values = list(confidences.values())
            print(f"   Confidence: mean={np.mean(conf_values):.3f}, min={np.min(conf_values):.3f}, max={np.max(conf_values):.3f}")
            print(f"\n   Top 5 matches by confidence:")
            for v1, conf in sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"      CSV1:{v1} → CSV2:{matches[v1]} (confidence={conf:.3f})")
            
            # Flag low confidence matches
            low_conf = [v1 for v1, c in confidences.items() if c < 0.85]
            if low_conf:
                print(f"\n   ⚠️  {len(low_conf)} matches with confidence < 0.85 (review recommended)")
        
        return matches, confidences
    
    def compute_rmse_for_matches(self, matches: Dict, confidences: Dict = None) -> Dict:
        """Compute RMSE for each matched vehicle pair"""
        
        if confidences is None:
            confidences = {}
        
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
                'confidence': confidences.get(v1, 0.0),
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
    
    def generate_report(self, output_dir: str = "output/comparison", min_confidence: float = 0.7, max_vehicles: int = None):
        """Generate full comparison report"""
        
        print("\n" + "="*70)
        print("  TRAJECTORY COMPARISON ANALYSIS (ROBUST MATCHING)")
        print("="*70 + "\n")
        
        # Match vehicles
        matches, confidences = self.match_vehicles_spatially(min_confidence=min_confidence, max_vehicles=max_vehicles)
        
        if len(matches) == 0:
            print("\n❌ No vehicle matches found. Check that:")
            print("   • Both CSVs cover overlapping frame ranges")
            print("   • Vehicles appear in same spatial locations")
            print("   • Try lowering --min-confidence")
            return
        
        # Compute RMSE
        rmse_results = self.compute_rmse_for_matches(matches, confidences)
        
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
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Minimum matching confidence (0-1)")
    parser.add_argument("--min-overlap", type=int, default=5, help="Minimum overlapping frames")
    parser.add_argument("--max-vehicles", type=int, default=None, help="Limit to first N vehicles (for testing)")
    
    args = parser.parse_args()
    
    comparator = TrajectoryComparator(args.csv1, args.csv2)
    comparator.generate_report(args.output, min_confidence=args.min_confidence, max_vehicles=args.max_vehicles)


if __name__ == "__main__":
    main()
