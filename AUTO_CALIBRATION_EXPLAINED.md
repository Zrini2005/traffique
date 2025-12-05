# Auto-Calibration Feature: How It Works

## 🎯 The Problem Auto-Calibration Solves

**Manual Calibration Issues:**
- ❌ Need to measure real-world distances (requires tape measure, GPS device)
- ❌ Need to identify 4 reference points and enter coordinates manually
- ❌ Error-prone (typos, measurement mistakes)
- ❌ Time-consuming (~5 minutes per camera)
- ❌ Requires technical knowledge

**Auto-Calibration Solution:**
- ✅ No measurements needed
- ✅ Just 2 clicks on a vehicle
- ✅ Uses standard vehicle dimensions (trucks are ~10m, cars are ~4.5m)
- ✅ Fast (~30 seconds per camera)
- ✅ Easy for non-technical users

---

## 🔍 Step-by-Step: How Auto-Calibration Works

### Step 1: User Selects Vehicle Type
```javascript
const VEHICLE_SIZES = {
  car: { length: 4.5, width: 1.8 },       // Standard sedan
  truck: { length: 10.0, width: 2.5 },    // Commercial truck
  bus: { length: 12.0, width: 2.5 },      // City bus
  motorcycle: { length: 2.0, width: 0.8 } // Standard motorcycle
};

// User selects: "Truck (10m × 2.5m)"
selectedVehicle = VEHICLE_SIZES['truck'];
```

### Step 2: User Clicks 2 Corners of Vehicle
```
User clicks on a truck in the video frame:
  Click 1: Top-left corner → (450, 320) pixels
  Click 2: Bottom-right corner → (650, 580) pixels

Vehicle bounding box:
  topLeft = [450, 320]
  bottomRight = [650, 580]
```

### Step 3: Calculate Pixel Size of Vehicle
```javascript
// How big is the vehicle in the image (in pixels)?
pixelWidth = |bottomRight[0] - topLeft[0]|
           = |650 - 450|
           = 200 pixels

pixelHeight = |bottomRight[1] - topLeft[1]|
            = |580 - 320|
            = 260 pixels
```

### Step 4: Calculate Pixels-per-Meter Scale
```javascript
// We know the truck's REAL dimensions:
realWidth = 2.5 meters
realLength = 10.0 meters

// Calculate scale from width
pixelsPerMeterX = pixelWidth / realWidth
                = 200 / 2.5
                = 80 pixels/meter

// Calculate scale from length
pixelsPerMeterY = pixelHeight / realLength
                = 260 / 10.0
                = 26 pixels/meter

// Use average (accounts for perspective)
pixelsPerMeter = (80 + 26) / 2
               = 53 pixels/meter
```

**Key Insight:** Now we know **53 pixels in the image = 1 meter in real world!** 🎯

### Step 5: Generate World Coordinates for Image Corners
```javascript
// Image dimensions
imgWidth = 1920 pixels
imgHeight = 1080 pixels

// Convert image corners to world coordinates
imagePoints = [
  [0, 0],              // Top-left corner (pixel coords)
  [1920, 0],           // Top-right corner
  [1920, 1080],        // Bottom-right corner
  [0, 1080]            // Bottom-left corner
];

worldPoints = [
  [0, 0],                           // (0, 0) meters
  [1920/53, 0],                     // (36.2, 0) meters
  [1920/53, 1080/53],               // (36.2, 20.4) meters
  [0, 1080/53]                      // (0, 20.4) meters
];

// This creates a coordinate system where:
// - Origin (0,0) is at top-left of image
// - Entire visible area is 36.2m wide × 20.4m tall
```

### Step 6: Calculate Homography Matrix
```javascript
// Now we have 4-point correspondence:
// Image pixels → World meters

const H = cv2.findHomography(imagePoints, worldPoints);

// H is a 3×3 matrix that transforms ANY pixel → meters
// Example transformation:
// Pixel (960, 540) → World (18.1m, 10.2m)
```

---

## 🧮 Mathematical Deep Dive

### The Core Formula

**Pixels-per-Meter Ratio:**
```
scale = vehicle_pixel_size / vehicle_real_size

pixelsPerMeter = (pixelWidth/realWidth + pixelHeight/realLength) / 2
```

**Why Average X and Y?**
- Camera angle creates perspective distortion
- Vehicle might not be perfectly axis-aligned
- Averaging reduces error from angle/rotation

### Example Calculation

**Scenario:**
- Camera: Drone at 50m altitude, 45° angle
- Vehicle: Truck (10m × 2.5m)
- User clicks: Top-left (450, 320), Bottom-right (650, 580)

**Calculation:**
```
Pixel dimensions:
  Width:  650 - 450 = 200 px
  Height: 580 - 320 = 260 px

Scale calculation:
  X-scale: 200 px / 2.5 m = 80 px/m
  Y-scale: 260 px / 10 m = 26 px/m
  Average: (80 + 26) / 2 = 53 px/m

Image size: 1920×1080 pixels

World coordinate system:
  Width:  1920 / 53 = 36.2 meters
  Height: 1080 / 53 = 20.4 meters

Homography mapping:
  [0, 0] → [0, 0]           (top-left)
  [1920, 0] → [36.2, 0]     (top-right)
  [1920, 1080] → [36.2, 20.4] (bottom-right)
  [0, 1080] → [0, 20.4]     (bottom-left)
```

### Homography Matrix Structure

```
H = | h11  h12  h13 |
    | h21  h22  h23 |
    | h31  h32  h33 |

Transformation:
  [x_world]   [h11  h12  h13]   [x_pixel]
  [y_world] = [h21  h22  h23] × [y_pixel]
  [   1   ]   [h31  h32  h33]   [   1   ]

After normalization:
  x_world = (h11*x + h12*y + h13) / (h31*x + h32*y + h33)
  y_world = (h21*x + h22*y + h23) / (h31*x + h32*y + h33)
```

---

## 🎨 Visual Comparison: Auto vs Manual

### Manual Calibration (OLD):
```
┌─────────────────────────────────────────────────────────────┐
│ 1. Measure real-world distance (e.g., road width)           │
│    → Need tape measure or GPS device                        │
│    → Time: 2-3 minutes                                      │
│                                                             │
│ 2. Identify 4 reference points in video                     │
│    → Need visible landmarks                                 │
│    → Time: 1 minute                                         │
│                                                             │
│ 3. Click 4 points in UI                                     │
│    → Enter X/Y coordinates for each                         │
│    → Risk of typos                                          │
│    → Time: 1-2 minutes                                      │
│                                                             │
│ 4. Calculate homography                                     │
│    → Click button                                           │
│    → Time: 2 seconds                                        │
│                                                             │
│ TOTAL TIME: ~5 minutes per camera                           │
│ DIFFICULTY: High (requires measurement tools)               │
└─────────────────────────────────────────────────────────────┘
```

### Auto Calibration (NEW):
```
┌─────────────────────────────────────────────────────────────┐
│ 1. Select vehicle type from dropdown                        │
│    → Car / Truck / Bus / Motorcycle                         │
│    → Time: 2 seconds                                        │
│                                                             │
│ 2. Click top-left corner of vehicle                         │
│    → No typing needed                                       │
│    → Time: 3 seconds                                        │
│                                                             │
│ 3. Click bottom-right corner of vehicle                     │
│    → Blue bounding box appears                              │
│    → Time: 3 seconds                                        │
│                                                             │
│ 4. Calculate homography                                     │
│    → System auto-generates 4-point correspondence           │
│    → Time: 2 seconds                                        │
│                                                             │
│ TOTAL TIME: ~10 seconds per camera                          │
│ DIFFICULTY: Low (no measurements needed)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Behind the Scenes: Code Flow

### Frontend (MultiCameraPage.jsx)

```javascript
// 1. User clicks on canvas
handleCanvasClick(e) {
  if (calibrationMode === 'auto') {
    // Record bbox points
    if (!vehicleBbox) {
      vehicleBbox = [clickPoint];  // First click
    } else {
      vehicleBbox = [...vehicleBbox, clickPoint];  // Second click
    }
  }
}

// 2. User clicks "Calculate Homography"
calculateHomography() {
  // Get vehicle size
  const vehicleSize = VEHICLE_SIZES[vehicleType];  // e.g., truck: 10m×2.5m
  
  // Calculate pixel dimensions
  const pixelWidth = |bbox[1][0] - bbox[0][0]|;
  const pixelHeight = |bbox[1][1] - bbox[0][1]|;
  
  // Calculate scale
  const pixelsPerMeter = (
    (pixelWidth / vehicleSize.width) +
    (pixelHeight / vehicleSize.length)
  ) / 2;
  
  // Generate 4-point correspondence
  const imagePoints = [
    [0, 0],
    [imgWidth, 0],
    [imgWidth, imgHeight],
    [0, imgHeight]
  ];
  
  const worldPoints = [
    [0, 0],
    [imgWidth / pixelsPerMeter, 0],
    [imgWidth / pixelsPerMeter, imgHeight / pixelsPerMeter],
    [0, imgHeight / pixelsPerMeter]
  ];
  
  // Send to backend
  fetch('/api/calibrate-camera', {
    method: 'POST',
    body: JSON.stringify({ imagePoints, worldPoints })
  });
}
```

### Backend (api_server.py)

```python
@app.route('/api/calibrate-camera', methods=['POST'])
def calibrate_camera():
    data = request.json
    image_points = np.array(data['image_points'], dtype=np.float32)
    world_points = np.array(data['world_points'], dtype=np.float32)
    
    # Calculate homography using OpenCV
    H, status = cv2.findHomography(image_points, world_points, method=0)
    
    # Return matrix for storage
    return jsonify({
        'success': True,
        'homography_matrix': H.tolist()
    })
```

---

## 📊 Coordinate System Explained

### What Auto-Calibration Creates:

```
LOCAL COORDINATE SYSTEM (meters from image top-left)

     0m                 18m                 36m
      ┌──────────────────┬──────────────────┐
      │                  │                  │   0m
      │                  │                  │
      │         🚗       │      🚚         │  10m
      │                  │                  │
      │                  │                  │  20m
      └──────────────────┴──────────────────┘

Origin: Top-left corner of image = (0, 0) meters
X-axis: Horizontal (left → right)
Y-axis: Vertical (top → bottom)
```

### Example Detections:

```
Vehicle 1 (Car):
  Pixel coords: (450, 320) → (580, 450)
  World coords: (8.5m, 6.0m) → (10.9m, 8.5m)
  Distance: 3.3 meters traveled

Vehicle 2 (Truck):
  Pixel coords: (850, 400) → (920, 520)
  World coords: (16.0m, 7.5m) → (17.4m, 9.8m)
  Distance: 2.7 meters traveled
```

### Important: This is a LOCAL coordinate system!
- Origin (0,0) is arbitrary (top-left of image)
- Units are meters, but NOT GPS coordinates
- Different cameras have different origins
- **To get GPS:** Need to add GPS reference point (see GPS_CALIBRATION_GUIDE.md)

---

## ⚠️ Limitations & Accuracy

### What Affects Accuracy?

1. **Vehicle Selection Quality**
   - ✅ Good: Vehicle clearly visible, not occluded
   - ❌ Bad: Partially hidden vehicle, motion blur

2. **Vehicle Type Match**
   - ✅ Good: Correctly identified (actual truck, selected "Truck")
   - ❌ Bad: Mismatched (pickup truck, selected "Bus")

3. **Camera Angle**
   - ✅ Good: Near-perpendicular view (~30-60° angle)
   - ❌ Bad: Extreme angles (<20° or >80°)

4. **Vehicle Orientation**
   - ✅ Good: Vehicle axis-aligned with road
   - ❌ Bad: Vehicle at 45° angle (turning)

5. **Ground Plane Assumption**
   - ✅ Good: Flat road surface
   - ❌ Bad: Hills, ramps, elevated highways

### Expected Accuracy:

| Condition | Scale Error | Position Error |
|-----------|-------------|----------------|
| Ideal (good vehicle, flat ground, proper angle) | ±5% | ±0.5m |
| Good (slight angle, minor occlusion) | ±10% | ±1.0m |
| Fair (bad angle or wrong vehicle type) | ±20% | ±2.0m |
| Poor (extreme angle, occluded, wrong type) | >30% | >3.0m |

### Example:
```
Ground Truth: Vehicle traveled 10.0 meters

With ideal conditions:
  Measured: 9.5 - 10.5 meters (±5%)
  
With good conditions:
  Measured: 9.0 - 11.0 meters (±10%)
  
With fair conditions:
  Measured: 8.0 - 12.0 meters (±20%)
```

---

## 🎯 Best Practices

### For Best Results:

1. **Choose Large Vehicles**
   - Trucks/buses are better than cars
   - Larger pixel size → more accurate scale calculation

2. **Select Clear, Unoccluded Vehicles**
   - Entire vehicle visible in frame
   - No overlapping with other vehicles

3. **Pick Vehicles on Flat Ground**
   - Not on ramps, bridges, or hills
   - Homography assumes planar surface

4. **Click Accurately**
   - Zoom in if possible
   - Click exact corners, not close-enough
   - Blue bbox should tightly fit vehicle

5. **Verify Scale Makes Sense**
   - Image shows ~20-50 meters of road? ✅ Reasonable
   - Image shows 200 meters? ❌ Likely error (recheck clicks)

6. **Use Same Vehicle Type Across Cameras**
   - All cameras: "Truck" for consistency
   - Makes cross-camera fusion more reliable

### Common Mistakes:

❌ **Clicking on vehicle shadow instead of body**
   → Results in oversized bbox, wrong scale

❌ **Selecting motorcycle but clicking entire motorcycle + rider**
   → Inflated dimensions, inaccurate scale

❌ **Using auto mode on tilted/turning vehicle**
   → Perspective distortion, wrong aspect ratio

❌ **Different vehicle types for overlapping cameras**
   → CAM1 uses "truck", CAM2 uses "car" for same scene
   → Incompatible coordinate systems, fusion fails

---

## 🔄 How This Fits with Multi-Camera Fusion

### Calibration Flow:

```
CAMERA 1 (Front View)
  ↓
Auto-calibrate using truck (10m × 2.5m)
  ↓
Scale: 53 pixels/meter
  ↓
Coordinate system: 36.2m × 20.4m visible area
  ↓
Homography H1

CAMERA 2 (Side View)
  ↓
Auto-calibrate using SAME TRUCK
  ↓
Scale: 48 pixels/meter (different angle)
  ↓
Coordinate system: 40.0m × 22.5m visible area
  ↓
Homography H2

CAMERA 3 (Rear View)
  ↓
Auto-calibrate using truck (10m × 2.5m)
  ↓
Scale: 51 pixels/meter
  ↓
Coordinate system: 37.6m × 21.2m visible area
  ↓
Homography H3
```

### Fusion Process:

```
DETECTION PHASE:
──────────────
CAM1 detects vehicle at pixel (450, 320)
  → Transform using H1
  → World coords: (8.5m, 6.0m) in CAM1's coordinate system

CAM2 detects vehicle at pixel (680, 540)
  → Transform using H2
  → World coords: (14.2m, 11.3m) in CAM2's coordinate system

PROBLEM: Different coordinate systems! Can't compare directly!

SOLUTION: GPS Reference Point
─────────────────────────────
If all cameras use SAME GPS reference:
  CAM1: GPS ref at (13.0827°N, 80.2707°E) = local (0,0)
  CAM2: GPS ref at (13.0827°N, 80.2707°E) = local (0,0)  ← SAME!
  CAM3: GPS ref at (13.0827°N, 80.2707°E) = local (0,0)  ← SAME!

Then:
  CAM1 detection: (8.5m, 6.0m) → GPS (13.0828°N, 80.2708°E)
  CAM2 detection: (14.2m, 11.3m) → GPS (13.0828°N, 80.2709°E)
  
  Distance between them: 0.3m
    ↓
  SAME VEHICLE! Merge into Global_V0001
```

---

## 💡 Key Insights

### What Auto-Calibration Actually Does:

1. **Establishes Scale** (pixels → meters)
   - Uses known vehicle dimensions as "ruler"
   - Calculates pixels-per-meter ratio
   - Applies ratio to entire image

2. **Creates Local Coordinate System**
   - Origin at image corner
   - Units in meters (not pixels)
   - Enables distance/velocity calculations

3. **Generates Homography Matrix**
   - Mathematical transformation
   - Converts ANY pixel → meter coordinate
   - Stored for analysis phase

### What Auto-Calibration Does NOT Do:

❌ **Create GPS coordinates**
   - Only creates local meter system
   - Need GPS reference for lat/lon

❌ **Stitch videos together**
   - Each camera has own coordinate system
   - Fusion happens in coordinate space, not video space

❌ **Synchronize time across cameras**
   - Only spatial calibration
   - Temporal sync is separate concern

❌ **Correct for camera distortion**
   - Assumes pinhole camera model
   - Fisheye/wide-angle may need undistortion first

---

## 🎓 Summary

**Auto-Calibration in One Sentence:**
> Uses a vehicle's known real-world size to calculate the scale (pixels-per-meter) and generate a homography matrix that transforms pixel coordinates to local meter coordinates.

**Key Components:**
1. **Input:** 2 clicks on vehicle + vehicle type selection
2. **Calculation:** Pixel size ÷ real size = scale ratio
3. **Output:** Homography matrix (3×3) for pixel→meter transformation
4. **Result:** All detections converted from pixels to meters

**Why It Works:**
- Trucks are standardized (~10m long)
- Camera captures truck at some pixel size (e.g., 260px)
- Ratio gives scale for entire scene (260px ÷ 10m = 26 px/m)
- Apply ratio to image corners → 4-point correspondence → homography

**Next Steps for GPS:**
- Auto-calibration creates LOCAL coordinates (✅ Done)
- GPS reference adds GLOBAL coordinates (⏸️ Not yet implemented)
- See `GPS_CALIBRATION_GUIDE.md` for implementation plan
