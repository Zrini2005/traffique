# Trajectory Accuracy Metrics - RMSE & Precision-Recall

## Overview
Generate RMSE and Precision-Recall curves to evaluate trajectory tracking accuracy against ground truth.

---

## Quick Start

### Step 1: Create Ground Truth Annotations
Manually annotate vehicle positions in a few frames for validation:

```powershell
conda activate iitmlab

# Annotate specific frames
python create_ground_truth.py --video "C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4" --output "output/ground_truth.csv" --frames 100,500,1000,1500,2000

# Or auto-select every 100th frame (10 frames total)
python create_ground_truth.py --video "C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4" --output "output/ground_truth.csv" --stride 100 --count 10
```

**Instructions:**
- Left-click on each vehicle's center
- Press `n` to move to next frame
- Press `u` to undo last point
- Press `s` to save and exit

---

### Step 2: Run Accuracy Metrics
Compare your predictions against ground truth:

```powershell
python trajectory_accuracy_metrics.py --csv "output/trajectories/d1f1_trajectories.csv" --gt "output/ground_truth.csv" --output "output/metrics"
```

---

## Outputs

### 1. RMSE Analysis (`rmse_analysis.png`)
Four subplots showing:
- **Error Distribution Histogram** - spread of position errors
- **Cumulative Distribution** - what % of points have error < X
- **Error Over Time** - temporal trends in accuracy
- **Per-Vehicle RMSE** - which vehicles are hardest to track

### 2. Precision-Recall Curve (`precision_recall_curve.png`)
Two plots:
- **PR Curve** - precision vs recall at different distance thresholds
- **F1 vs Threshold** - optimal threshold for detection matching

### 3. Metrics JSON (`accuracy_metrics.json`)
Numerical results:
```json
{
  "rmse_metrics": {
    "rmse_euclidean": 1.234,
    "mae_euclidean": 0.987,
    "rmse_x": 0.876,
    "rmse_y": 0.943,
    "max_error": 5.432,
    "median_error": 0.765,
    "std_error": 0.654
  },
  "detection_metrics": {
    "precision": 0.923,
    "recall": 0.887,
    "f1_score": 0.905,
    "total_tp": 1234,
    "total_fp": 102,
    "total_fn": 157
  },
  "best_operating_point": {
    "best_threshold": 2.5,
    "best_f1": 0.912,
    "best_precision": 0.931,
    "best_recall": 0.893
  }
}
```

---

## Ground Truth CSV Format

The ground truth CSV should match the prediction format:

```csv
vehicle_id,frame,x_px,y_px
1,100,450,320
2,100,680,410
1,101,455,322
...
```

Optional columns (if using meters):
```csv
vehicle_id,frame,x_px,y_px,x_m,y_m
1,100,450,320,20.45,14.54
...
```

---

## Interpretation Guide

### RMSE Metrics
- **RMSE < 2m** → Excellent tracking accuracy
- **RMSE 2-5m** → Good (acceptable for traffic analysis)
- **RMSE > 5m** → Poor (needs improvement)

### Detection Metrics
- **Precision** → % of predicted vehicles that are real
  - High precision = few false alarms
- **Recall** → % of real vehicles that were detected
  - High recall = few missed vehicles
- **F1 Score** → Harmonic mean (balances precision & recall)
  - F1 > 0.9 = excellent
  - F1 0.7-0.9 = good
  - F1 < 0.7 = needs improvement

### Distance Threshold
- Controls how close a prediction must be to ground truth to count as correct
- **2m threshold** = strict (good for safety-critical apps)
- **5m threshold** = lenient (OK for aggregate traffic stats)
- Optimal threshold maximizes F1 score

---

## Presentation Tips

### For Slides
1. Show **Error Distribution** histogram
   - Highlight mean/median error
   - Compare to baseline or other methods

2. Show **Precision-Recall Curve**
   - Point out operating point
   - Compare to industry benchmarks (if available)

3. Show **Key Numbers**
   ```
   RMSE:      1.23 m  ✓
   Precision: 92.3%   ✓
   Recall:    88.7%   ✓
   F1 Score:  90.5%   ✓
   ```

### For Demo
1. Show ground truth annotation process (quick video)
2. Show output graphs side-by-side
3. Explain what each metric means in plain language

---

## Advanced Usage

### Compare Different Smoothing Methods
Test raw vs smoothed trajectories:

```powershell
# Export raw trajectories (modify trajectory_tracker.py to skip smoothing)
python trajectory_tracker.py ... --no-smooth --csv "output/raw_trajectories.csv"

# Compare
python trajectory_accuracy_metrics.py --csv "output/raw_trajectories.csv" --gt "output/ground_truth.csv" --output "output/metrics_raw"
python trajectory_accuracy_metrics.py --csv "output/smooth_trajectories.csv" --gt "output/ground_truth.csv" --output "output/metrics_smooth"
```

### Subset Analysis
Test on specific vehicle types or time windows:

```python
import pandas as pd
pred = pd.read_csv("output/trajectories.csv")
gt = pd.read_csv("output/ground_truth.csv")

# Only frames 1000-2000
pred_subset = pred[(pred['frame'] >= 1000) & (pred['frame'] <= 2000)]
gt_subset = gt[(gt['frame'] >= 1000) & (gt['frame'] <= 2000)]

pred_subset.to_csv("output/pred_subset.csv", index=False)
gt_subset.to_csv("output/gt_subset.csv", index=False)

# Run metrics on subset
# python trajectory_accuracy_metrics.py --csv output/pred_subset.csv --gt output/gt_subset.csv
```

---

## Troubleshooting

**Problem:** "No matching trajectories found"
- Check that vehicle_id and frame columns match between CSVs
- Ensure annotated frames overlap with tracked frames

**Problem:** RMSE seems too high
- Check scale calibration (meters_per_pixel)
- Verify ground truth annotations are accurate
- Consider if detection/tracking has systematic bias

**Problem:** Low precision
- Too many false detections → increase confidence threshold
- Check if SAHI is creating duplicate detections

**Problem:** Low recall
- Missing vehicles → lower confidence threshold
- Increase SAHI overlap or use smaller slice size

---

## Citation

If presenting this work, acknowledge:
- VisDrone dataset for pre-trained models
- SAHI (Slicing Aided Hyper Inference) library
- FilterPy (Kalman filter implementation)
- Scikit-learn (precision-recall metrics)
