#!/usr/bin/env python3
"""
Trajectory Accuracy Metrics - RMSE & Precision-Recall Curves

Compares predicted trajectories against ground truth annotations.
Generates performance graphs for presentation/analysis.

Usage:
  python trajectory_accuracy_metrics.py --csv <trajectory_csv> --gt <ground_truth_csv> --output <output_dir>
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns

sns.set_style("whitegrid")


class TrajectoryAccuracyAnalyzer:
    """Compute accuracy metrics for trajectory predictions"""
    
    def __init__(self, predictions_csv: str, ground_truth_csv: str):
        """
        Load predictions and ground truth
        
        Ground truth CSV should have columns:
          - vehicle_id, frame, x_px (or x_m), y_px (or y_m)
        Predictions CSV from trajectory_tracker.py already has this format
        """
        self.pred_df = pd.read_csv(predictions_csv)
        self.gt_df = pd.read_csv(ground_truth_csv)
        
        # Determine if using meters or pixels
        self.use_meters = 'x_m' in self.pred_df.columns and 'x_m' in self.gt_df.columns
        self.x_col = 'x_m' if self.use_meters else 'x_px'
        self.y_col = 'y_m' if self.use_meters else 'y_px'
        self.unit = 'm' if self.use_meters else 'px'
        
        print(f"Loaded predictions: {len(self.pred_df)} points")
        print(f"Loaded ground truth: {len(self.gt_df)} points")
        print(f"Using {'meters' if self.use_meters else 'pixels'} for measurements")
    
    def align_trajectories(self, distance_threshold: float = None) -> pd.DataFrame:
        """
        Align predictions with ground truth by spatial matching within each frame.
        Uses nearest-neighbor matching since vehicle IDs won't correspond.
        """
        
        if distance_threshold is None:
            distance_threshold = 50.0 if not self.use_meters else 5.0
        
        from scipy.spatial.distance import cdist
        
        aligned_rows = []
        
        # Process each frame separately
        for frame_no in self.gt_df['frame'].unique():
            gt_frame = self.gt_df[self.gt_df['frame'] == frame_no]
            pred_frame = self.pred_df[self.pred_df['frame'] == frame_no]
            
            if len(pred_frame) == 0:
                continue
            
            gt_points = gt_frame[[self.x_col, self.y_col]].values
            pred_points = pred_frame[[self.x_col, self.y_col]].values
            
            # Compute distance matrix
            dist_matrix = cdist(pred_points, gt_points, metric='euclidean')
            
            # Match each GT point to nearest prediction
            matched_preds = set()
            for gt_idx in range(len(gt_points)):
                # Find closest prediction to this GT point
                pred_distances = dist_matrix[:, gt_idx]
                closest_pred_idx = np.argmin(pred_distances)
                min_dist = pred_distances[closest_pred_idx]
                
                # Only match if within threshold and not already matched
                if min_dist <= distance_threshold and closest_pred_idx not in matched_preds:
                    matched_preds.add(closest_pred_idx)
                    
                    # Create aligned row
                    gt_row = gt_frame.iloc[gt_idx]
                    pred_row = pred_frame.iloc[closest_pred_idx]
                    
                    aligned_row = {
                        'frame': frame_no,
                        'vehicle_id_gt': gt_row['vehicle_id'],
                        'vehicle_id_pred': pred_row['vehicle_id'],
                        f'{self.x_col}_gt': gt_row[self.x_col],
                        f'{self.y_col}_gt': gt_row[self.y_col],
                        f'{self.x_col}_pred': pred_row[self.x_col],
                        f'{self.y_col}_pred': pred_row[self.y_col],
                        'match_distance': min_dist
                    }
                    aligned_rows.append(aligned_row)
        
        aligned = pd.DataFrame(aligned_rows)
        print(f"Aligned {len(aligned)} trajectory points (matched by position)")
        print(f"  • Matching threshold: {distance_threshold:.1f} {self.unit}")
        print(f"  • Frames with matches: {aligned['frame'].nunique()}")
        
        return aligned
    
    def compute_rmse(self, aligned_df: pd.DataFrame) -> Dict[str, float]:
        """Compute RMSE metrics"""
        
        # Extract coordinates
        x_pred = aligned_df[f'{self.x_col}_pred'].values
        y_pred = aligned_df[f'{self.y_col}_pred'].values
        x_gt = aligned_df[f'{self.x_col}_gt'].values
        y_gt = aligned_df[f'{self.y_col}_gt'].values
        
        # Euclidean distance error
        errors = np.sqrt((x_pred - x_gt)**2 + (y_pred - y_gt)**2)
        
        # Per-axis errors
        x_errors = np.abs(x_pred - x_gt)
        y_errors = np.abs(y_pred - y_gt)
        
        metrics = {
            'rmse_euclidean': np.sqrt(np.mean(errors**2)),
            'mae_euclidean': np.mean(errors),
            'rmse_x': np.sqrt(np.mean(x_errors**2)),
            'rmse_y': np.sqrt(np.mean(y_errors**2)),
            'max_error': np.max(errors),
            'median_error': np.median(errors),
            'std_error': np.std(errors)
        }
        
        return metrics, errors
    
    def compute_detection_metrics(self, distance_threshold: float = 2.0) -> Dict:
        """
        Compute precision/recall for detections
        
        A predicted point is considered a true positive if it's within
        distance_threshold of a ground truth point in the same frame.
        """
        
        # Group by frame
        results = []
        
        for frame_no in self.gt_df['frame'].unique():
            gt_frame = self.gt_df[self.gt_df['frame'] == frame_no]
            pred_frame = self.pred_df[self.pred_df['frame'] == frame_no]
            
            gt_points = gt_frame[[self.x_col, self.y_col]].values
            pred_points = pred_frame[[self.x_col, self.y_col]].values
            
            if len(pred_points) == 0:
                # No predictions - all ground truth are false negatives
                results.append({
                    'frame': frame_no,
                    'tp': 0,
                    'fp': 0,
                    'fn': len(gt_points),
                    'n_gt': len(gt_points),
                    'n_pred': 0
                })
                continue
            
            if len(gt_points) == 0:
                # No ground truth - all predictions are false positives
                results.append({
                    'frame': frame_no,
                    'tp': 0,
                    'fp': len(pred_points),
                    'fn': 0,
                    'n_gt': 0,
                    'n_pred': len(pred_points)
                })
                continue
            
            # Compute distance matrix
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(pred_points, gt_points, metric='euclidean')
            
            # Match predictions to ground truth (greedy nearest neighbor)
            matched_gt = set()
            tp = 0
            
            for pred_idx in range(len(pred_points)):
                min_dist_idx = np.argmin(dist_matrix[pred_idx])
                min_dist = dist_matrix[pred_idx, min_dist_idx]
                
                if min_dist <= distance_threshold and min_dist_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(min_dist_idx)
            
            fp = len(pred_points) - tp
            fn = len(gt_points) - tp
            
            results.append({
                'frame': frame_no,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'n_gt': len(gt_points),
                'n_pred': len(pred_points)
            })
        
        results_df = pd.DataFrame(results)
        
        # Overall metrics
        total_tp = results_df['tp'].sum()
        total_fp = results_df['fp'].sum()
        total_fn = results_df['fn'].sum()
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'threshold': distance_threshold
        }
        
        return metrics, results_df
    
    def plot_rmse_analysis(self, aligned_df: pd.DataFrame, errors: np.ndarray, 
                          output_dir: str = "output/metrics"):
        """Generate RMSE visualization plots"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Error distribution histogram
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Histogram
        axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(np.mean(errors), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(errors):.3f} {self.unit}')
        axes[0, 0].axvline(np.median(errors), color='green', linestyle='--',
                          label=f'Median: {np.median(errors):.3f} {self.unit}')
        axes[0, 0].set_xlabel(f'Position Error ({self.unit})')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Position Error Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        axes[0, 1].plot(sorted_errors, cumulative, linewidth=2)
        axes[0, 1].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(0.95, color='orange', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel(f'Position Error ({self.unit})')
        axes[0, 1].set_ylabel('Cumulative Probability')
        axes[0, 1].set_title('Cumulative Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error over time
        axes[1, 0].plot(aligned_df['frame'], errors, alpha=0.5, linewidth=0.5)
        axes[1, 0].plot(aligned_df['frame'].rolling(50).mean(), 
                       pd.Series(errors).rolling(50).mean(), 
                       color='red', linewidth=2, label='50-frame moving avg')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel(f'Position Error ({self.unit})')
        axes[1, 0].set_title('Error Over Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Per-vehicle RMSE
        vehicle_rmse = aligned_df.groupby('vehicle_id').apply(
            lambda g: np.sqrt(np.mean(
                (g[f'{self.x_col}_pred'] - g[f'{self.x_col}_gt'])**2 +
                (g[f'{self.y_col}_pred'] - g[f'{self.y_col}_gt'])**2
            ))
        ).sort_values(ascending=False).head(20)
        
        axes[1, 1].barh(range(len(vehicle_rmse)), vehicle_rmse.values)
        axes[1, 1].set_yticks(range(len(vehicle_rmse)))
        axes[1, 1].set_yticklabels([f'V{vid}' for vid in vehicle_rmse.index])
        axes[1, 1].set_xlabel(f'RMSE ({self.unit})')
        axes[1, 1].set_title('Top 20 Vehicles by RMSE')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_path / 'rmse_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  📊 Saved: rmse_analysis.png")
        plt.close()
    
    def plot_precision_recall_curve(self, output_dir: str = "output/metrics"):
        """Generate Precision-Recall curve at different distance thresholds"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Compute metrics at various thresholds
        thresholds = np.linspace(0.5, 10, 30) if self.use_meters else np.linspace(5, 100, 30)
        
        precisions = []
        recalls = []
        f1_scores = []
        
        print("\nComputing precision-recall at different thresholds...")
        for thresh in thresholds:
            metrics, _ = self.compute_detection_metrics(distance_threshold=thresh)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1_scores.append(metrics['f1_score'])
        
        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Precision-Recall curve
        axes[0].plot(recalls, precisions, marker='o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall Curve\n(Different Distance Thresholds)')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1.05])
        axes[0].set_ylim([0, 1.05])
        
        # Add threshold annotations at key points
        for i in [0, len(thresholds)//2, -1]:
            axes[0].annotate(f'{thresholds[i]:.1f}{self.unit}',
                           xy=(recalls[i], precisions[i]),
                           xytext=(10, -10), textcoords='offset points',
                           fontsize=9, alpha=0.7)
        
        # F1 score vs threshold
        axes[1].plot(thresholds, f1_scores, marker='o', linewidth=2, markersize=4, color='green')
        axes[1].axvline(thresholds[np.argmax(f1_scores)], color='red', linestyle='--',
                       label=f'Best F1 @ {thresholds[np.argmax(f1_scores)]:.1f}{self.unit}')
        axes[1].set_xlabel(f'Distance Threshold ({self.unit})')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('F1 Score vs Distance Threshold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        print(f"  📊 Saved: precision_recall_curve.png")
        plt.close()
        
        # Return best threshold
        best_idx = np.argmax(f1_scores)
        return {
            'best_threshold': thresholds[best_idx],
            'best_f1': f1_scores[best_idx],
            'best_precision': precisions[best_idx],
            'best_recall': recalls[best_idx]
        }
    
    def generate_report(self, output_dir: str = "output/metrics"):
        """Generate comprehensive accuracy report"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("  TRAJECTORY ACCURACY ANALYSIS")
        print("="*70 + "\n")
        
        # Align trajectories
        aligned_df = self.align_trajectories()
        
        if len(aligned_df) == 0:
            print("❌ No matching trajectories found between predictions and ground truth!")
            return
        
        # RMSE metrics
        print("📐 Computing RMSE metrics...")
        rmse_metrics, errors = self.compute_rmse(aligned_df)
        
        print(f"\n  Position Error Metrics ({self.unit}):")
        print(f"    • RMSE (Euclidean): {rmse_metrics['rmse_euclidean']:.4f}")
        print(f"    • MAE (Euclidean):  {rmse_metrics['mae_euclidean']:.4f}")
        print(f"    • RMSE X-axis:      {rmse_metrics['rmse_x']:.4f}")
        print(f"    • RMSE Y-axis:      {rmse_metrics['rmse_y']:.4f}")
        print(f"    • Max Error:        {rmse_metrics['max_error']:.4f}")
        print(f"    • Median Error:     {rmse_metrics['median_error']:.4f}")
        print(f"    • Std Dev:          {rmse_metrics['std_error']:.4f}")
        
        # Detection metrics
        print(f"\n🎯 Computing detection metrics...")
        default_thresh = 2.0 if self.use_meters else 20.0
        det_metrics, det_df = self.compute_detection_metrics(distance_threshold=default_thresh)
        
        print(f"\n  Detection Metrics (threshold={default_thresh}{self.unit}):")
        print(f"    • Precision: {det_metrics['precision']:.4f}")
        print(f"    • Recall:    {det_metrics['recall']:.4f}")
        print(f"    • F1 Score:  {det_metrics['f1_score']:.4f}")
        print(f"    • True Positives:  {det_metrics['total_tp']}")
        print(f"    • False Positives: {det_metrics['total_fp']}")
        print(f"    • False Negatives: {det_metrics['total_fn']}")
        
        # Generate plots
        print(f"\n📊 Generating visualizations...")
        self.plot_rmse_analysis(aligned_df, errors, output_dir)
        pr_best = self.plot_precision_recall_curve(output_dir)
        
        print(f"\n  Best Operating Point:")
        print(f"    • Threshold: {pr_best['best_threshold']:.2f} {self.unit}")
        print(f"    • F1 Score:  {pr_best['best_f1']:.4f}")
        print(f"    • Precision: {pr_best['best_precision']:.4f}")
        print(f"    • Recall:    {pr_best['best_recall']:.4f}")
        
        # Save metrics to JSON
        import json
        report = {
            'rmse_metrics': rmse_metrics,
            'detection_metrics': det_metrics,
            'best_operating_point': pr_best,
            'dataset_info': {
                'n_predictions': len(self.pred_df),
                'n_ground_truth': len(self.gt_df),
                'n_aligned': len(aligned_df),
                'unit': self.unit
            }
        }
        
        with open(output_path / 'accuracy_metrics.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✅ Report saved to: {output_path}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory accuracy metrics (RMSE, Precision-Recall)"
    )
    parser.add_argument("--csv", required=True, help="Predicted trajectories CSV")
    parser.add_argument("--gt", required=True, help="Ground truth trajectories CSV")
    parser.add_argument("--output", default="output/metrics", help="Output directory")
    
    args = parser.parse_args()
    
    analyzer = TrajectoryAccuracyAnalyzer(args.csv, args.gt)
    analyzer.generate_report(args.output)


if __name__ == "__main__":
    main()
