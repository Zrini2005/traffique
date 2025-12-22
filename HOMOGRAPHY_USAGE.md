# Homography Calibration & Usage Guide

## Overview

Homography transforms **image pixel coordinates** to **real-world ground coordinates**, accounting for perspective distortion from the drone camera angle. This is essential for accurate trajectory comparison.

---

## 🔍 Why Homography?

Your friend's `D2F1_lclF.csv` uses homography:
- **X_pixel scale variation**: 0.36–3.59 m/px (151% coefficient of variation)
- **Y_pixel scale variation**: 0.0013–0.0086 m/px (51% coefficient of variation)
- A simple `meters_per_pixel` scale would have up to **10x error** across the frame

---

## 📋 Two Options

### Option 1: Use Existing Homography from `camera_calibration.json`

**✅ Use this if:**
- Your video is from the same camera setup as `D2F1_lclF.csv`
- Cameras were calibrated with ground control points

**Available matrices:**
```
D1_to_D2
D2_to_D3
D3_to_D4
D4_to_D5
```

**Determine which camera:**
1. Check your video filename: `D1F1_stab.mp4` → likely **D1** or **D2**
2. Run a test to see which produces sensible world coordinates

**Test with D1_to_D2:**
```powershell
python extract_homography.py camera_calibration.json D1_to_D2 --output output/homography_d1_to_d2.json

python trajectory_tracker.py "C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4" `
  --start 0 --frames 200 `
  --format d2f1 `
  --homography output/homography_d1_to_d2.json `
  --csv output/test_d1_to_d2.csv `
  --output output/test_d1_to_d2
```

**Check if world coordinates match:**
```powershell
python -c "import pandas as pd; df = pd.read_csv('output/test_d1_to_d2.csv'); print(f'X_world: {df['X_world'].min():.1f} to {df['X_world'].max():.1f}'); print(f'Y_world: {df['Y_world'].min():.1f} to {df['Y_world'].max():.1f}')"
```

Expected ranges (from `D2F1_lclF.csv`):
- **X_world**: 1052.9 – 1255.1 meters
- **Y_world**: 0.1 – 8.6 meters

If ranges are wildly different (e.g., negative or > 10000), try another matrix.

---

### Option 2: Compute New Homography from Scratch

**✅ Use this if:**
- You don't know which camera matrix to use
- Camera positions changed
- You want independent calibration

**Steps:**

1. **Extract a frame with clear reference points:**
   ```powershell
   python save_video_frame.py "C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4" --frame 5000 --output output/calibration_frame.png
   ```

2. **Run calibration tool (interactive):**
   ```powershell
   python compute_homography.py `
     --video "C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4" `
     --frame 5000 `
     --output output/my_homography.json
   ```

   **Instructions during calibration:**
   - Click on known points (lane markings, crosswalks, building corners)
   - Enter their real-world X, Y coordinates in meters
   - Collect **at least 4 points** (more is better)
   - Press `s` to save, `u` to undo last point

   **Where to get world coordinates?**
   - Use Google Earth satellite view
   - Measure from maps with known scales
   - Use survey data if available
   - Coordinate system: choose origin (e.g., bottom-left of road), X = along road, Y = across road

3. **Run tracking with your homography:**
   ```powershell
   python trajectory_tracker.py "C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4" `
     --start 0 --frames -1 `
     --format d2f1 `
     --homography output/my_homography.json `
     --csv output/my_trajectories_d2f1.csv `
     --output output/my_trajectories
   ```

---

## 🚀 Full Pipeline Workflow

### Step 1: Determine Camera & Extract Matrix
```powershell
# Extract D1_to_D2 homography
python extract_homography.py camera_calibration.json D1_to_D2 --output output/homography_d1_to_d2.json
```

### Step 2: Run Full Video Tracking with Homography
```powershell
python trajectory_tracker.py "C:\Users\sakth\Documents\traffique_footage\D1F1_stab.mp4" `
  --start 0 --frames -1 `
  --format d2f1 `
  --homography output/homography_d1_to_d2.json `
  --csv output/my_d2f1_with_homography.csv `
  --output output/homography_results `
  --min-length 15
```

### Step 3: Compare with Friend's Trajectory
```powershell
python compare_trajectories.py `
  --csv1 output/my_d2f1_with_homography.csv `
  --csv2 output/D2F1_lclF.csv `
  --output output/final_comparison `
  --distance-threshold 5.0
```

This will:
- Match vehicles spatially (despite ID differences)
- Compute RMSE in world coordinates
- Generate comparison plots

---

## 📊 Expected Output

With homography applied:
- **Scale variation** should drop from 151% to <5%
- **X_world, Y_world** should match friend's ranges
- **RMSE** between your and friend's trajectories should be low (<1 meter for good tracking)

---

## 🛠️ Troubleshooting

### "Homography produces negative or extreme coordinates"
→ Wrong camera matrix; try another (D2_to_D3, etc.) or compute new homography

### "World coordinates don't match friend's ranges"
→ Check coordinate system origin; may need to translate/rotate after homography

### "Reprojection error > 1 meter"
→ Need more accurate ground control points or better-distributed points

---

## 📂 Files Created

| File | Purpose |
|------|---------|
| `compute_homography.py` | Interactive calibration tool |
| `extract_homography.py` | Extract matrix from camera_calibration.json |
| `output/homography_*.json` | Homography matrix files |
| `trajectory_tracker.py` | **Updated** with `--homography` argument |

---

## ✅ Validation Checklist

- [ ] Homography loaded successfully
- [ ] World coordinate ranges match expected (1000–1300 X, 0–10 Y)
- [ ] Scale variation < 10% across frame
- [ ] RMSE with friend's data < 2 meters
- [ ] Visual trajectory comparison looks similar

---

## 🎯 Next Steps

1. **Test which camera matrix works** (D1_to_D2, D2_to_D3, etc.)
2. **Run full video tracking** with correct homography
3. **Compare results** with `D2F1_lclF.csv`
4. **Validate accuracy** using RMSE and visual plots
