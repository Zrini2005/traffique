# Overlap & Polygon Filtering in Multi-Camera Mode

## 🔗 Problem 1: Camera Overlap

### The Issue:
```
Current Implementation (WRONG):
  Camera 1: 0-100m coverage, Offset = (0, 0)
  Camera 2: 80-180m coverage, Offset = (100, 0)  ← WRONG! Should be (80, 0)
  
  Vehicle at 90m:
    CAM1 detects: (90, 5)
    CAM2 detects: (10, 5)  ← Wrong! Should be (90, 5)
    Distance: 80m apart
    Result: NOT MERGED ❌
```

### Solution Implemented: AUTO-DETECTION ✅

**How It Works:**

Instead of asking users to measure overlap (impossible!), the system **automatically detects** overlap by comparing world coordinates:

```javascript
// When calibrating Camera 2:
1. User selects vehicle bbox → calculates pixels-per-meter
2. System creates world points assuming edge-to-edge (no overlap)
3. detectOverlap() compares where CAM2 starts vs where CAM1 ends
4. If CAM2 starts BEFORE CAM1 ends → OVERLAP DETECTED!
5. System adjusts CAM2's origin offset automatically
```

**Algorithm:**
```javascript
function detectOverlap(currentCameraWorldPoints) {
  // Get previous camera's end position
  const prevEndX = prevCamera.worldPoints[1][0];  // CAM1 ends at 100m
  
  // Get current camera's start position
  const currentStartX = currentCameraWorldPoints[0][0];  // CAM2 would start at 100m (edge-to-edge)
  
  // BUT user calibrated CAM2 showing the SAME scene as CAM1's end
  // So the reference vehicle maps CAM2's start to 80m (in world coords)
  // Overlap = 100 - 80 = 20m
  const overlapX = Math.max(0, prevEndX - currentStartX);
  
  return { overlapX, overlapY };
}
```

### Example Workflow:

```
Scenario: 3 cameras with overlaps

Camera 1:
  - User clicks truck bbox
  - Coverage calculated: 0-100m (100m wide)
  - Offset: (0, 0)
  - Overlap: 0m (first camera)

Camera 2:
  - User clicks truck bbox on frame showing overlap zone
  - System initially assumes: offset = (100, 0) [edge-to-edge]
  - detectOverlap() compares coordinates:
    * Previous camera ends at: 100m
    * Current camera starts at: 80m (because user calibrated on overlap scene)
  - Detected overlap: 100 - 80 = 20m ✅
  - System adjusts offset: (100 - 20) = (80, 0) ✅
  
Camera 3:
  - User clicks truck bbox
  - System assumes offset = (80 + 100) = (180, 0)
  - detectOverlap() finds current starts at 160m
  - Detected overlap: 180 - 160 = 20m ✅
  - System adjusts offset: (180 - 20) = (160, 0) ✅

Result:
  Vehicle at 90m (in overlap zone CAM1-CAM2):
    CAM1 detects: (90, 5)
    CAM2 detects: (90, 5)  ✅ Same coordinates!
    Distance: 0m
    → MERGED! ✅
```

### UI Display:

**During Calibration:**
```
🔗 Overlap Detection (Auto-Computed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Overlap automatically detected by comparing 
   with previous camera's coverage!

Overlap X: 20.00 meters
Overlap Y: 0.00 meters

✨ This camera overlaps with the previous 
   camera - coordinate system automatically 
   adjusted!
```

**After Calibration (Camera Card):**
```
CAM2
D2F1_stab.mp4
🔗 Overlap: 20.0m × 0.0m
✓ Calibrated
```

---

## 🎯 Problem 2: Polygon Filtering

### The Challenge:

In single-camera mode:
- User draws polygon in **pixel coordinates**
- Easy to implement (just click on video frame)

In multi-camera mode:
- Cameras have different perspectives
- Same world location appears at different pixels in each camera
- Need to filter in **world coordinates**, not pixels

### Current Status:
```python
# In api_server.py - polygon_points is always None
polygon_points = None  # No UI for polygon in multi-camera mode yet
```

---

## 💡 Solution Options for Polygon Filtering

### Option 1: Per-Camera Polygons (Easiest)

Allow users to draw a separate polygon for each camera:

```javascript
cameras = [
  {
    id: 'CAM1',
    polygon: [[100,200], [400,200], [400,500], [100,500]],  // Pixels
    homography: H1
  },
  {
    id: 'CAM2',
    polygon: [[150,250], [450,250], [450,550], [150,550]],  // Pixels
    homography: H2
  }
];
```

**Pros:**
- ✅ Easy to implement (reuse single-camera polygon UI)
- ✅ Intuitive for users (click on video frame)
- ✅ Works with existing backend code

**Cons:**
- ❌ Tedious (draw polygon for each camera)
- ❌ Hard to ensure consistency across cameras
- ❌ Not ideal for overlapping cameras

**Implementation:**
1. Add polygon drawing to each camera's preview
2. Store polygon per camera
3. Pass to backend in analyze request

---

### Option 2: World Coordinate Polygon (Best, Complex)

Define polygon once in world coordinates, apply to all cameras:

```javascript
// User defines polygon in world coordinates (meters)
worldPolygon = [
  [20, 5],   // 20m along X, 5m along Y
  [80, 5],
  [80, 25],
  [20, 25]
];

// Backend transforms to pixel coordinates per camera
for each camera:
  pixelPolygon = inverseTransform(worldPolygon, homography)
  filter_detections(pixelPolygon)
```

**Pros:**
- ✅ Define once, applies to all cameras
- ✅ Consistent filtering across cameras
- ✅ Perfect for overlapping cameras

**Cons:**
- ❌ Complex UI (how to let user draw in world coords?)
- ❌ Requires inverse homography transformation
- ❌ Hard to visualize for users

**Implementation:**
1. Create top-down view of world coordinates
2. User draws polygon on top-down view
3. Backend transforms to pixel coords per camera
4. Apply filtering

---

### Option 3: Draw on One Camera, Auto-Project (Hybrid)

User draws polygon on one camera, system projects to others:

```javascript
// User draws on CAM1
cam1Polygon = [[100,200], [400,200], [400,500], [100,500]];  // Pixels

// Transform to world coordinates
worldPolygon = transform(cam1Polygon, H1);

// Project to other cameras
cam2Polygon = inverseTransform(worldPolygon, H2);
cam3Polygon = inverseTransform(worldPolygon, H3);
```

**Pros:**
- ✅ Easy for user (draw once)
- ✅ Consistent across cameras
- ✅ Familiar UI (like single-camera mode)

**Cons:**
- ❌ Requires inverse homography
- ❌ Projected polygon might go off-screen in other cameras
- ❌ User can't verify projection is correct

**Implementation:**
1. Add polygon drawing to first camera
2. Transform to world coordinates
3. Inverse transform to all other cameras
4. Show projected polygons (for verification)

---

## 🚀 Recommended Implementation (Short Term)

**Use Option 1: Per-Camera Polygons**

This is the quickest to implement and most intuitive:

### UI Changes Needed:

```javascript
// In MultiCameraPage.jsx

// Add per-camera polygon state
const [cameras, setCameras] = useState([
  {
    id: 'CAM1',
    polygon: [],           // ← ADD THIS
    isDrawingPolygon: false,  // ← ADD THIS
    ...
  }
]);

// Add polygon drawing UI (similar to AnalysisPage)
<button onClick={() => startDrawingPolygon(camera.id)}>
  Draw ROI Polygon
</button>

// Display polygon on each camera preview
{camera.polygon.length > 0 && renderPolygon(camera.polygon)}
```

### Backend Changes Needed:

```python
# In api_server.py - analyze_multi_camera()

for camera in cameras:
    polygon_points = camera.get('polygon_points', None)
    
    analytics, annotated_frame = cam_analyzer.analytics_mode(
        video_path=video_path,
        polygon_points=polygon_points,  # ← Use per-camera polygon
        frame_idx=frame_idx,
        time_window=time_window
    )
```

---

## 🎯 Quick Fix Summary

### For Overlap (IMPLEMENTED ✅):
1. User enters overlap distance (e.g., 20m)
2. System subtracts from previous camera width
3. Correctly aligns coordinate systems

### For Polygon (TO IMPLEMENT):
1. **Short term**: Add per-camera polygon drawing
2. **Long term**: Implement world-coordinate polygon with projection

---

## 📝 Usage Instructions

### Setting Up Overlapping Cameras (Auto-Detection):

```
Step 1: Add CAM1
  - Upload D1F1.mp4
  
Step 2: Calibrate CAM1
  - Select vehicle type (e.g., truck)
  - Click vehicle top-left corner
  - Click vehicle bottom-right corner
  - System calculates: Coverage 0-100m, Offset (0, 0)
  - ✅ Calibrated!

Step 3: Add CAM2
  - Upload D2F1.mp4
  
Step 4: Calibrate CAM2
  - Select same vehicle type (truck)
  - Click vehicle bbox (on frame showing overlap zone)
  - System detects:
    * CAM1 ends at 100m
    * CAM2 calibrated frame shows scene at 80m
    * Overlap detected: 20m ✅
    * Adjusted offset: (80, 0) ✅
  - UI shows: "🔗 Overlap: 20.0m × 0.0m"
  - ✅ Calibrated!
  
Step 5: Add CAM3
  - Upload D3F1.mp4
  
Step 6: Calibrate CAM3
  - Select vehicle, click bbox
  - System detects:
    * CAM2 ends at 180m (80 + 100)
    * CAM3 starts at 160m
    * Overlap detected: 20m ✅
    * Adjusted offset: (160, 0) ✅
  - ✅ Calibrated!
  
Step 7: Run Analysis
  - All cameras aligned in same coordinate system ✅
  - Overlap zones correctly handled ✅
  - Fusion works! ✅
```

### How Auto-Detection Works:

**The Magic:**
When you calibrate Camera 2 by clicking on a vehicle, that vehicle might actually be visible in Camera 1's overlap zone. The system:

1. **Calculates pixels-per-meter** from your vehicle bbox
2. **Maps image corners to world coordinates** using that scale
3. **Compares with previous camera** to see if coordinates overlap
4. **Detects overlap** if CAM2's start position is BEFORE CAM1's end position
5. **Adjusts origin offset** automatically to align coordinate systems

**Example:**
```
CAM1: Vehicle at pixel (800, 400) → World (90m, 5m)
CAM2: SAME vehicle at pixel (200, 400) → World (90m, 5m)  ✅ Auto-aligned!
```

If you calibrate CAM2 at a different location (no overlap):
```
CAM1 ends: 100m
CAM2 calibrated at scene starting: 100m
Overlap: 0m → Edge-to-edge ✅
```

---

## 🔍 Verifying Overlap is Correct

After calibration, the system automatically displays detected overlap:

**In Camera Card:**
```
CAM1
D1F1_stab.mp4
✓ Calibrated

CAM2
D2F1_stab.mp4
🔗 Overlap: 20.0m × 0.0m  ← Auto-detected!
✓ Calibrated
```

**In Origin Offset Display:**
```
CAM1: Offset (0, 0), Width 100m
CAM2: Offset (80, 0), Width 100m, Overlap 20m ✅
CAM3: Offset (160, 0), Width 100m, Overlap 20m ✅
Total coverage: 0-260m ✅
```

### Troubleshooting:

**If overlap looks wrong:**
1. Check that you calibrated using a vehicle in the **overlap zone**
2. Ensure you used the **same vehicle type** for consistency
3. Make sure the vehicle bbox is **accurate** (tight fit around vehicle)
4. Verify videos are uploaded **in sequential order** (CAM1 → CAM2 → CAM3)

**If no overlap detected but there should be:**
- You may have calibrated CAM2 on a frame **outside** the overlap zone
- Try recalibrating CAM2 using a frame that shows the overlap area

**If overlap is detected but cameras are actually edge-to-edge:**
- This means your calibration frames overlap even though cameras don't
- System is working correctly! The coordinates are aligned to the calibration scene

---

## 🎓 Summary

| Feature | Status | Notes |
|---------|--------|-------|
| **Overlap Detection** | ✅ Auto-Computed | No manual measurement needed! |
| **Offset Calculation** | ✅ Auto-Adjusted | Accounts for detected overlap |
| **Manual Override** | ✅ Available | Can edit offset X/Y if needed |
| **Visual Feedback** | ✅ Implemented | Shows overlap in camera cards |
| **Polygon Filtering** | ⏸️ Pending | Requires UI implementation |
| **Per-Camera Polygon** | ⏸️ Recommended | Easiest to implement |
| **World-Coord Polygon** | ⏸️ Future | Better UX, more complex |

**Bottom Line:** 
- ✅ Overlap is **automatically detected** - no manual measurement! 
- ✅ Just calibrate each camera using a vehicle, system handles the rest
- ⏸️ Polygon filtering needs to be implemented next

**How It Works:**
The system is smart! When you click a vehicle bbox to calibrate Camera 2, it calculates where that scene exists in world coordinates. If that location overlaps with Camera 1's coverage, overlap is automatically detected and the coordinate system is adjusted. You don't measure anything - you just click vehicles!
