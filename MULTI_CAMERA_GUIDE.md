# Multi-Camera Mode Quick Guide

## 🎯 Overview

The **Advanced Mode** allows you to analyze multiple drone/camera videos with overlapping coverage areas. The system:
- Calibrates each camera to real-world coordinates
- Detects vehicles in each camera independently  
- Fuses detections across cameras using spatial matching
- Outputs unified tracks in meters (not pixels)

## 🚀 Step-by-Step Usage

### **Step 1: Upload Videos**
1. Click **"Advanced Mode"** toggle in the header
2. Add camera videos using one of two methods:
   - **Upload Video Files**: For smaller videos (< 500MB) - drag and drop or browse
   - **Use Local Videos**: For large videos (e.g., D1F1_stab.mp4) - select from C:\Users\sakth\Documents\traffique_footage
3. Videos will be automatically processed
4. Each camera gets an ID (CAM1, CAM2, etc.)

**Pro Tip**: Use local videos for large drone footage to avoid long upload times!

### **Step 2: Calibrate Cameras**
For each camera, you need to map image coordinates to real-world meters.

**Choose Your Calibration Method:**

#### **🚀 Auto Calibration (Recommended - Easiest!)**
1. Click **"Calibrate"** button for a camera
2. Select **"Auto (Recommended)"** mode
3. Choose vehicle type from dropdown:
   - **Car** (4.5m × 1.8m)
   - **Truck** (10m × 2.5m) ← Default
   - **Bus** (12m × 2.5m)
   - **Motorcycle** (2m × 0.8m)
4. Click **top-left corner** of a vehicle in the image
5. Click **bottom-right corner** of the same vehicle
6. System automatically calculates scale based on vehicle size!
7. Click **"Calculate Homography"**

**Why Auto is Better:**
- ✅ No manual measurements needed
- ✅ Just 2 clicks per camera
- ✅ Uses standard vehicle dimensions
- ✅ Fast and accurate
- ✅ No math required!

#### **📍 Manual Calibration (Advanced)**
For when you need precise control or have measured reference points:

1. Click **"Calibrate"** button for a camera
2. Select **"Manual"** mode
3. **Select 4 points** on the video frame that form a rectangle in the real world
   - Good examples: Road corners, lane markings, building corners
   - Points should be clearly visible and at ground level
4. **Enter real-world coordinates** for each point in meters
   - Measure the actual distance in meters
   - Example: A 50m x 30m road section
   ```
   Point 1: (0, 0)     Point 2: (50, 0)
   Point 3: (50, 30)   Point 4: (0, 30)
   ```
5. Click **"Calculate Homography"**
6. Repeat for all cameras

**⚠️ CRITICAL: Upload Cameras in Sequential Order!**

For sequential/overlapping cameras, **you MUST add videos in order** (CAM1 → CAM2 → CAM3):

```
✅ CORRECT ORDER:
  1. Add D1F1.mp4 as CAM1 (covers 0-100m)
  2. Add D2F1.mp4 as CAM2 (covers 80-180m)
  3. Add D3F1.mp4 as CAM3 (covers 160-260m)
  4. Calibrate CAM1 first → System sets offset (0, 0)
  5. Calibrate CAM2 next → System auto-calculates offset based on CAM1's coverage
  6. Calibrate CAM3 last → System auto-calculates offset based on CAM1+CAM2

Result: All cameras aligned in same coordinate system ✅

❌ WRONG ORDER:
  1. Add D3F1.mp4 as CAM1 (rear camera first)
  2. Add D1F1.mp4 as CAM2 (front camera second)
  3. Add D2F1.mp4 as CAM3 (middle camera last)
  
Result: Coordinate systems misaligned, fusion fails ❌
```

**How Auto-Calculation Works:**

When you calibrate a camera, the system:
1. Looks at all **previously calibrated** cameras
2. Calculates their coverage width/height (from worldPoints)
3. Sums up the coverage as cumulative offset
4. Applies that offset to the current camera's origin

**Example:**
```
Camera 1 Calibration:
  - Auto-calibration using truck (10m × 2.5m)
  - Scale: 50 px/m
  - Image: 1920×1080 px
  - Coverage: 1920/50 = 38.4m wide, 1080/50 = 21.6m tall
  - Origin offset: (0, 0) [first camera]
  - World coordinates: (0, 0) to (38.4m, 21.6m)

Camera 2 Calibration:
  - Auto-calibration using truck (10m × 2.5m)
  - Scale: 48 px/m
  - Image: 1920×1080 px
  - Coverage: 1920/48 = 40m wide, 1080/48 = 22.5m tall
  - Origin offset: (38.4, 0) [AUTO-CALCULATED from CAM1's width!]
  - World coordinates: (38.4, 0) to (78.4m, 22.5m)

Camera 3 Calibration:
  - Auto-calibration using truck (10m × 2.5m)
  - Scale: 52 px/m
  - Image: 1920×1080 px
  - Coverage: 1920/52 = 36.9m wide, 1080/52 = 20.8m tall
  - Origin offset: (78.4, 0) [AUTO-CALCULATED from CAM1+CAM2 widths!]
  - World coordinates: (78.4, 0) to (115.3m, 20.8m)

Total road coverage: 0m to 115.3m ✅
```

**Pro Tips:**
- **Auto Mode**: Pick a vehicle that's clearly visible and roughly perpendicular to camera
- **Auto Mode**: Trucks and buses work best (larger, easier to mark accurately)
- **Auto Mode**: Vehicle should be on flat ground, not on a slope
- **Auto Mode**: **Origin offset is AUTO-CALCULATED** - just calibrate cameras in order!
- **Auto Mode**: You can manually override offset if needed (advanced users only)
- **Manual Mode**: Use a consistent coordinate system across all cameras
- **Manual Mode**: Pick points with known real-world distances
- **Manual Mode**: Avoid points on moving objects or far from the road
- The 4 points (manual) should form a rectangle/trapezoid when projected

**⚠️ What if I added cameras in wrong order?**

If you accidentally added cameras out of order:
1. Remove all cameras (Trash icon)
2. Re-add cameras in correct sequential order
3. Calibrate in order (CAM1 first, then CAM2, then CAM3)

**⚠️ What about overlapping cameras with different angles?**

For overlapping cameras viewing the SAME road section from different angles:
- If they fully overlap: All use offset (0, 0) - manual override needed
- If they partially overlap: Add in sequence, system will auto-calculate
- Mixed scenario: Use manual offset override in calibration UI

### **Step 3: Set Parameters**
- **Time Window**: How many seconds to track vehicles (10-120s)
- **SAHI**: Enable for better small vehicle detection (slower)

### **Step 4: Process**
1. Review your settings
2. Click **"Start Multi-Camera Analysis"**
3. Wait for processing (estimated time shown)
4. Backend will:
   - Detect vehicles in each camera view
   - Transform coordinates to world space (meters)
   - Fuse overlapping detections
   - Assign global vehicle IDs

### **Step 5: View Results**
- **Total Vehicles**: Unique vehicles across all cameras
- **Cross-Camera Tracks**: Vehicles seen in multiple cameras
- **Camera-wise Results**: Annotated frames from each camera
- **Download CSV**: Fused results with real-world coordinates

## 📊 Output CSV Format

```csv
global_id,cameras_seen,world_x_start,world_y_start,world_x_end,world_y_end,velocity_mps,distance_m,time_in_scene,trajectory_world
V0001,"[CAM1,CAM2]",10.5,20.3,45.2,18.7,12.5,35.8,2.86,"[[10.5,20.3],[15.2,19.5],...,[45.2,18.7]]"
V0002,"[CAM2]",5.2,10.1,50.3,12.4,15.2,45.5,3.00,"[[5.2,10.1],[10.3,10.8],...,[50.3,12.4]]"
```

**Columns:**
- `global_id`: Unique vehicle ID across all cameras
- `cameras_seen`: Which cameras detected this vehicle
- `world_x_start/end, world_y_start/end`: Position in meters
- `velocity_mps`: Speed in meters per second
- `distance_m`: Total distance traveled in meters
- `time_in_scene`: How long vehicle was tracked
- `trajectory_world`: Full path in real-world coordinates

## 🎓 Calibration Example

**Scenario**: You have a 50m x 30m road section visible in Camera 1

1. **Click 4 corners of the road**:
   - Top-left corner
   - Top-right corner (50m away)
   - Bottom-right corner
   - Bottom-left corner (30m from top)

2. **Enter coordinates**:
   ```
   Point 1: X=0,  Y=0   (origin)
   Point 2: X=50, Y=0   (50m right)
   Point 3: X=50, Y=30  (50m right, 30m down)
   Point 4: X=0,  Y=30  (30m down)
   ```

3. **Calculate**: System creates transformation matrix

4. **Result**: All detections in Camera 1 now in meters!

## 🔧 Troubleshooting

**"Calibration failed"**
- Ensure 4 points form a valid quadrilateral
- Points should not be collinear
- Check world coordinates are reasonable

**"Low detection count"**
- Try enabling SAHI
- Check if vehicles are inside polygon ROI
- Increase time window

**"Cross-camera tracks is 0"**
- Cameras may not have overlapping coverage
- Check calibration coordinates use same reference system
- Increase spatial threshold in fusion (edit `spatial_threshold` in `api_server.py`)

**Processing takes too long**
- Disable SAHI for faster processing
- Reduce time window
- Use fewer cameras

## 🎯 Best Practices

1. **Camera Setup**:
   - Use cameras with overlapping coverage (20-30% overlap ideal)
   - Similar frame rates across cameras
   - Good lighting conditions

2. **Calibration**:
   - Measure real-world distances accurately
   - Use ground-level reference points
   - Consistent coordinate system across all cameras

3. **Performance**:
   - Start with 2-3 cameras
   - Use 30s time window for testing
   - Enable SAHI only if detection quality is poor

4. **Validation**:
   - Check annotated frames for detection quality
   - Verify cross-camera tracks make sense
   - Spot-check CSV coordinates are reasonable

## 📝 Comparison: Simple vs Advanced Mode

| Feature | Simple Mode | Advanced Mode |
|---------|-------------|---------------|
| Videos | 1 | 2-5 |
| Calibration | Not required | Required (4 points per camera) |
| Coordinates | Pixels | Meters (real-world) |
| Fusion | N/A | Spatial matching |
| Output | Single camera CSV | Unified multi-camera CSV |
| Complexity | Low | Medium |
| Use Case | Quick analysis | Research, accurate measurements |

---

**Ready to try?** Click **"Advanced Mode"** and upload your first multi-camera dataset! 🚀
