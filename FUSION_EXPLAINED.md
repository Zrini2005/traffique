# How Multi-Camera Fusion Works (Without Video Stitching)

## 🎯 Key Concept: Coordinate Transformation, NOT Video Stitching

The system **does NOT create a merged/stitched video**. Instead, it:
1. Processes each video **independently**
2. Transforms detections into a **common coordinate system** (real-world meters)
3. **Fuses duplicate detections** of the same vehicle seen by multiple cameras

---

## 📐 The Homography Matrix Role

### What It Does:
- **Transforms pixel coordinates → real-world coordinates**
- Maps each camera's perspective into a common "bird's-eye view" coordinate system
- Enables spatial comparison between cameras

### Example:
```
Camera 1 detects vehicle at:  (pixel_x=450, pixel_y=320)
                               ↓ [Apply Homography Matrix H1]
                               → (world_x=25.3m, world_y=10.8m)

Camera 2 detects vehicle at:  (pixel_x=680, pixel_y=540)
                               ↓ [Apply Homography Matrix H2]
                               → (world_x=25.5m, world_y=10.9m)

RESULT: System recognizes these are the SAME vehicle (only 0.28m apart)
```

---

## 🔄 Processing Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-CAMERA ANALYSIS                         │
└─────────────────────────────────────────────────────────────────┘

STEP 1: Individual Camera Processing
───────────────────────────────────
Camera 1 (D1F1.mp4)              Camera 2 (D2F1.mp4)              Camera 3 (D3F1.mp4)
     │                                │                                │
     ├─ Run YOLO detection            ├─ Run YOLO detection            ├─ Run YOLO detection
     ├─ Track vehicles                ├─ Track vehicles                ├─ Track vehicles
     ├─ Get pixel coords              ├─ Get pixel coords              ├─ Get pixel coords
     │                                │                                │
     └─► Detections in pixels         └─► Detections in pixels         └─► Detections in pixels
         Vehicle_C1_001: (450,320)        Vehicle_C2_003: (680,540)        Vehicle_C3_007: (200,400)
         Vehicle_C1_002: (800,450)        Vehicle_C2_004: (300,250)        Vehicle_C3_008: (550,380)


STEP 2: Coordinate Transformation
──────────────────────────────────
Apply Homography Matrix H to each detection
   
Camera 1 → World Coords          Camera 2 → World Coords          Camera 3 → World Coords
  (450,320) → (25.3m, 10.8m)       (680,540) → (25.5m, 10.9m)       (200,400) → (15.2m, 8.5m)
  (800,450) → (35.1m, 15.2m)       (300,250) → (15.1m, 8.6m)        (550,380) → (35.0m, 15.3m)


STEP 3: Spatial Fusion
──────────────────────
Group detections within 2m threshold

Cluster 1:                       Cluster 2:                       Cluster 3:
  (25.3m, 10.8m) ← Camera 1        (15.2m, 8.5m) ← Camera 3         (35.1m, 15.2m) ← Camera 1
  (25.5m, 10.9m) ← Camera 2        (15.1m, 8.6m) ← Camera 2         (35.0m, 15.3m) ← Camera 3
     ↓                               ↓                                ↓
  Global_V0001                    Global_V0002                     Global_V0003
  (seen by CAM1, CAM2)            (seen by CAM2, CAM3)             (seen by CAM1, CAM3)
  Merged coords: (25.4m, 10.85m)  Merged coords: (15.15m, 8.55m)   Merged coords: (35.05m, 15.25m)


STEP 4: Final Output
────────────────────
CSV with unified vehicle tracks:

global_id  | cameras_seen       | world_x_start | world_y_start | velocity_mps | class
-----------+--------------------+---------------+---------------+--------------+------
V0001      | ['CAM1', 'CAM2']   | 25.4          | 10.85         | 12.5         | car
V0002      | ['CAM2', 'CAM3']   | 15.15         | 8.55          | 8.3          | truck
V0003      | ['CAM1', 'CAM3']   | 35.05         | 15.25         | 15.2         | car
```

---

## 🎬 What You See vs What System Does

### ❌ What Does NOT Happen:
- Videos are **NOT** merged into a single video file
- No panoramic stitching like Google Photos
- No frame-by-frame video alignment
- No visual blending of camera feeds

### ✅ What DOES Happen:
- Each video processed **separately** (parallel processing possible)
- Detections extracted as **coordinate lists**
- Coordinates transformed to **common reference frame**
- **Duplicate detections merged** based on spatial proximity
- Output is **data**, not video (CSV with global vehicle IDs)

---

## 🔍 Example Scenario

### Setup:
- **Camera 1**: Front gate angle (see road segment A-B)
- **Camera 2**: Side angle (see road segment B-C, overlaps with Camera 1 at segment B)
- **Camera 3**: Rear angle (see road segment C-D)

### Detection:
A red car drives through the entire scene (A → B → C → D)

### System Behavior:

```
Timeline:
─────────────────────────────────────────────────────────────
Time: 0s-10s  │ Camera 1 sees car at segment A-B
              │ Detection: Vehicle_C1_005 (pixels)
              │ Transform: (30.5m, 12.0m) in world coords
              
Time: 8s-15s  │ Camera 1 sees car at segment B (overlap zone)
              │ Detection: Vehicle_C1_005 continues
              │ Camera 2 sees car at segment B (overlap zone)
              │ Detection: Vehicle_C2_012 (NEW in Camera 2)
              │ 
              │ Fusion: Both detections are within 2m threshold
              │ → Merged into Global_V0007
              │ → CSV shows: cameras_seen=['CAM1', 'CAM2']
              
Time: 13s-20s │ Camera 2 sees car at segment B-C
              │ Detection: Vehicle_C2_012 continues
              │ Camera 3 sees car at segment C
              │ Detection: Vehicle_C3_018 (NEW in Camera 3)
              │
              │ Fusion: Both detections are within 2m threshold
              │ → Merged into Global_V0007
              │ → CSV updated: cameras_seen=['CAM1', 'CAM2', 'CAM3']
              
Time: 18s-25s │ Camera 3 sees car at segment C-D
              │ Detection: Vehicle_C3_018 continues
              
Result:
───────
One continuous track: Global_V0007
- Seen by all 3 cameras
- Total distance: ~50m (segment A to D)
- Average velocity: 10.5 m/s (37.8 km/h)
- Complete trajectory from A to D
```

---

## 🧮 Mathematical Details

### Homography Transformation:
```python
# Input: pixel coordinates [x_px, y_px]
pixel_point = np.array([[[450, 320]]], dtype=np.float32)

# Homography matrix (3x3) calculated during calibration
H = np.array([
    [0.05, 0.01, -20.0],
    [0.02, 0.08, -15.0],
    [0.0001, 0.0002, 1.0]
])

# Transform to world coordinates [x_m, y_m]
world_point = cv2.perspectiveTransform(pixel_point, H)
# Result: [[25.3, 10.8]] meters
```

### Spatial Fusion Algorithm:
```python
# For each detection from Camera 1:
for det1 in camera1_detections:
    avg_x1 = (det1['world_x_start'] + det1['world_x_end']) / 2
    avg_y1 = (det1['world_y_start'] + det1['world_y_end']) / 2
    
    # Check all detections from Camera 2:
    for det2 in camera2_detections:
        avg_x2 = (det2['world_x_start'] + det2['world_x_end']) / 2
        avg_y2 = (det2['world_y_start'] + det2['world_y_end']) / 2
        
        # Calculate Euclidean distance
        distance = sqrt((avg_x1 - avg_x2)² + (avg_y1 - avg_y2)²)
        
        # If within threshold, merge
        if distance < 2.0:  # 2 meters
            merge_detections(det1, det2)
```

### Weighted Merging:
```python
# When merging, use weighted average based on tracking confidence
total_weight = det1['frames_tracked'] + det2['frames_tracked']

merged_x = (det1['world_x'] * det1['frames_tracked'] + 
            det2['world_x'] * det2['frames_tracked']) / total_weight

merged_y = (det1['world_y'] * det1['frames_tracked'] + 
            det2['world_y'] * det2['frames_tracked']) / total_weight
```

---

## 📊 Output Data Structure

### CSV Format:
```csv
global_id,cameras_seen,class,frames_tracked,world_x_start,world_y_start,world_x_end,world_y_end,velocity_mps,distance_m,time_in_scene
V0001,"['CAM1', 'CAM2']",car,85,25.4,10.85,45.2,20.3,12.5,23.8,1.904
V0002,"['CAM2', 'CAM3']",truck,120,15.15,8.55,35.0,18.2,8.3,22.1,2.663
V0003,"['CAM1']",motorcycle,42,5.2,3.1,15.8,8.5,15.6,11.9,0.763
```

### Key Fields:
- **global_id**: Unique ID across all cameras (V0001, V0002, ...)
- **cameras_seen**: List of cameras that detected this vehicle
- **world_x/y_start/end**: Real-world coordinates in **meters**
- **velocity_mps**: Speed in **meters per second** (not pixels!)
- **distance_m**: Total distance traveled in **meters**
- **frames_tracked**: Total frames across all cameras (confidence metric)

---

## 🎯 Benefits of This Approach

1. **No Video Storage**: Don't need to create massive stitched videos
2. **Parallel Processing**: Each camera can be processed independently
3. **Real-World Metrics**: Output in meters, not pixels
4. **Cross-Camera Tracking**: See which vehicles appear in multiple views
5. **Overlap Handling**: Vehicles in overlap zones counted once, not duplicated
6. **Scalable**: Add more cameras without exponential complexity

---

## 🚀 How to Use

1. **Calibrate each camera** (auto mode with vehicle, or manual mode with 4 points)
2. **Run multi-camera analysis** (system processes all cameras)
3. **Review fused results** (CSV with global IDs and real-world coordinates)
4. **Visualize trajectories** (future feature: overlay on map/satellite image)

---

## ❓ FAQ

**Q: Can I visualize the fused detections?**
A: Currently outputs CSV. Future feature: overlay trajectories on satellite map or bird's-eye view diagram.

**Q: What if cameras don't overlap?**
A: Works fine! Each detection gets its own global ID. Fusion only merges when cameras overlap.

**Q: How accurate is the 2m threshold?**
A: Depends on calibration accuracy. With proper calibration, ±0.5m is typical.

**Q: Can I adjust the fusion threshold?**
A: Yes, it's hardcoded as 2.0m in `fuse_detections()` function. You can change it.

**Q: Do cameras need to be synchronized?**
A: Not required! System uses spatial proximity, not temporal synchronization.

**Q: What about time differences between cameras?**
A: Currently ignored. Future enhancement: temporal filtering (only merge if timestamps are close).

---

## 🔧 Technical Limitations

1. **Static Cameras Only**: Assumes cameras don't move during recording
2. **Flat Ground Assumption**: Homography works best on flat surfaces
3. **No Temporal Sync**: Doesn't check if detections happen at same time
4. **Simple Clustering**: Uses basic spatial threshold (no advanced tracking algorithms like SORT/DeepSORT)
5. **No Occlusion Handling**: If vehicle hidden in one camera, tracking may break

---

## 🎓 Summary

**The system transforms the problem from:**
- ❌ "How do I stitch these videos together?" (complex, resource-intensive)

**To:**
- ✅ "How do I transform coordinates to a common reference frame?" (simple, efficient)

**Result:** Clean CSV data with unified vehicle tracks across all cameras, ready for analytics, visualization, or integration with other systems.
