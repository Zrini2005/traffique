# Advanced Trajectory Prediction System

A state-of-the-art trajectory prediction system for drone (Bird's Eye View) traffic footage using Transformer architecture with social attention.

## 🎯 Features

### Core Capabilities
- **Multi-Modal Prediction**: Predicts 3 possible future paths with confidence scores
- **Social Awareness**: Models vehicle interactions using Graph Attention Networks
- **Temporal Modeling**: Transformer-based encoder for motion pattern learning
- **Kalman Filtering**: State estimation for smooth trajectories
- **Uncertainty Quantification**: Multiple future modes with probabilities

### Technical Highlights
- **Architecture**: Transformer + Graph Neural Networks
- **Input**: Historical positions (20 frames = 0.8s at 25fps)
- **Output**: 30 future frames (1.2s) with 3 modes
- **Features**: Position, velocity, acceleration, social context
- **Metrics**: ADE, FDE, Miss Rate, Mode Selection Accuracy

---

## 📁 Project Structure

```
trajectory_prediction_system.py    # Core prediction model (Transformer)
predict_trajectories_advanced.py   # Integration with video pipeline
trajectory_evaluation.py           # Evaluation metrics (ADE, FDE, etc.)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Install PyTorch (choose your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install ultralytics sahi opencv-python-headless numpy
```

### 2. Basic Prediction

```python
from trajectory_prediction_system import TrajectoryPredictor
import numpy as np

# Initialize predictor
predictor = TrajectoryPredictor(
    history_length=20,
    future_length=30,
    num_modes=3
)

# Historical positions (20 time steps)
positions = np.array([[x, y] for x, y in your_trajectory])

# Predict future (returns 3 modes with probabilities)
future_modes, mode_probs = predictor.predict(
    positions,
    return_all_modes=True
)

print(f"Most likely mode: {np.argmax(mode_probs)}")
print(f"Confidence: {np.max(mode_probs):.1%}")
```

### 3. Video Analysis

```bash
# Analyze video and predict trajectories
python predict_trajectories_advanced.py \
    "/path/to/video.mp4" \
    --start 9700 \
    --frames 200 \
    --output output/predictions \
    --top-k 5
```

**Output**: Top 5 vehicles with:
- Historical path (green)
- 3 predicted modes (blue, orange, cyan)
- Probability for each mode
- Visualization overlaid on video frame

---

## 🏗️ System Architecture

### Phase 1: Data Preprocessing

```
Video → Stabilization → Detection (YOLO) → Tracking (IoU) → Tracklets
```

**Output**: Structured trajectories with positions over time

### Phase 2: Feature Engineering

```python
# For each vehicle:
positions:     (x, y)              # Pixel coordinates
velocities:    (vx, vy)            # Computed via finite differences
accelerations: (ax, ay)            # Second-order differences
neighbors:     [(x1,y1), (x2,y2)]  # Surrounding vehicles
```

### Phase 3: Model Architecture

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Historical Trajectory (20 frames)           │
│  [x, y, vx, vy, ax, ay] × 20                       │
└───────────────────┬─────────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │  Temporal Encoder   │
         │   (Transformer)     │
         │  • Self-attention   │
         │  • Position encode  │
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Social Encoder     │
         │  (Graph Attention)  │
         │  • Neighbor context │
         │  • Interaction model│
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  Multi-Modal        │
         │  Decoder (GRU)      │
         │  • 3 future modes   │
         │  • Mode probabilities│
         └──────────┬──────────┘
                    │
         ┌──────────▼──────────┐
         │  OUTPUT:            │
         │  • Mode 1 (60% prob)│
         │  • Mode 2 (30% prob)│
         │  • Mode 3 (10% prob)│
         └─────────────────────┘
```

### Phase 4: Prediction

**Multi-Modal Output**:
- **Mode 1**: Most likely path (e.g., continue straight)
- **Mode 2**: Alternative path (e.g., lane change left)
- **Mode 3**: Third option (e.g., lane change right)

Each mode has:
- 30 future (x, y) positions
- Confidence probability
- Cumulative displacement from current position

---

## 📊 Model Components

### 1. Temporal Encoder (Transformer)

```python
TemporalEncoder(
    input_dim=6,        # [x, y, vx, vy, ax, ay]
    hidden_dim=128,
    num_layers=3,
    num_heads=4
)
```

**Purpose**: Learn sequential patterns in vehicle motion
**Key Features**:
- Self-attention over time
- Positional encoding for temporal order
- Captures non-linear driving behaviors

### 2. Social Context Encoder (Graph Attention)

```python
SocialContextEncoder(
    feature_dim=64,
    num_heads=4
)
```

**Purpose**: Model vehicle interactions
**How it works**:
- Represents vehicles as graph nodes
- Edges encode proximity/relevance
- Attention weights learn influence patterns

### 3. Trajectory Decoder (GRU)

```python
TrajectoryDecoder(
    feature_dim=128,
    output_dim=2,
    num_modes=3
)
```

**Purpose**: Generate multiple future trajectories
**Output**:
- 3 autoregressive decoders (one per mode)
- Each predicts 30 future steps
- Mode probability classifier

---

## 📈 Evaluation Metrics

### Average Displacement Error (ADE)

```python
ADE = (1/T) Σ √((x̂_t - x_t)² + (ŷ_t - y_t)²)
```

**Meaning**: Average distance between predicted and actual path across all time steps

### Final Displacement Error (FDE)

```python
FDE = √((x̂_T - x_T)² + (ŷ_T - y_T)²)
```

**Meaning**: Distance between final predicted and actual positions (most important)

### Miss Rate

```python
MissRate = (# predictions with FDE > 2m) / (# total predictions) × 100%
```

**Threshold**: 2.0 meters (standard in autonomous driving)

### Mode Selection Accuracy

```python
Accuracy = (# times highest prob = best mode) / (# predictions) × 100%
```

**Purpose**: Evaluate if model assigns correct probabilities

---

## 🔧 Configuration

### Model Hyperparameters

```python
# History and prediction lengths
HISTORY_LENGTH = 20      # 0.8 seconds at 25fps
FUTURE_LENGTH = 30       # 1.2 seconds at 25fps

# Model architecture
HIDDEN_DIM = 128         # Feature dimension
NUM_LAYERS = 3           # Transformer layers
NUM_HEADS = 4            # Attention heads
NUM_MODES = 3            # Future trajectory modes

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
```

### Detection & Tracking

```python
# YOLO detection
MODEL = "yolov8x-visdrone"
CONFIDENCE = 0.25
USE_SAHI = True          # Sliced detection for small objects

# Tracking
MIN_IOU = 0.25           # Minimum IoU for matching
MAX_AGE = 25             # Max frames to keep lost tracks
```

---

## 🎓 Training Pipeline

### 1. Prepare Dataset

```python
from trajectory_evaluation import DatasetBuilder

builder = DatasetBuilder(
    history_length=20,
    future_length=30
)

# Build from trajectories
dataset = builder.build_dataset_from_trajectories(
    trajectories=your_trajectories,
    pixels_per_meter=20.0
)

# Split into train/val
train_data, val_data = builder.train_val_split(
    dataset,
    val_ratio=0.2
)
```

### 2. Train Model

```python
from trajectory_prediction_system import train_model

model = train_model(
    train_data=train_data,
    val_data=val_data,
    num_epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    save_path='models/trajectory_predictor.pth'
)
```

### 3. Evaluate

```python
from trajectory_evaluation import TrajectoryEvaluator

evaluator = TrajectoryEvaluator(miss_threshold=2.0)

metrics = evaluator.evaluate_dataset(
    predictions=model_predictions,
    ground_truths=ground_truth_trajectories
)

print(f"ADE: {metrics['ADE']:.2f}m")
print(f"FDE: {metrics['FDE']:.2f}m")
print(f"Miss Rate: {metrics['MissRate']:.1f}%")
```

---

## 🔬 Advanced Features

### 1. Coordinate Transformation

```python
from trajectory_evaluation import pixel_to_meters

# Convert pixel coordinates to physical meters
positions_meters = pixel_to_meters(
    positions_pixels,
    pixels_per_meter=20.0  # Calibration from ground truth
)
```

### 2. Kalman Filtering

```python
from utils.trajectory import smooth_kalman

# Smooth noisy detections
smoothed = smooth_kalman(
    positions,
    process_var=2.0,
    meas_var=5.0
)
```

### 3. Social Context

```python
# Automatically finds K-nearest neighbors
neighbor_positions = analyzer._get_neighbor_trajectories(
    ego_id=vehicle_id,
    ego_positions=positions,
    max_neighbors=5
)

# Uses relative positions for interaction modeling
```

---

## 📊 Performance Benchmarks

| Metric | Expected Value | State-of-the-Art |
|--------|---------------|------------------|
| **minADE** | 0.5 - 1.0m | 0.4m (Wayformer) |
| **minFDE** | 1.0 - 2.0m | 0.8m (Wayformer) |
| **Miss Rate** | 5 - 15% | 3% (Scene Transformer) |
| **Mode Accuracy** | 60 - 80% | 85% (VectorNet) |

*Note: Untrained model will have random performance. Training on drone traffic data required.*

---

## 🎨 Visualization Output

Each prediction image shows:

1. **Historical Path** (Green)
   - Solid line showing where vehicle traveled
   - Green circle at starting point

2. **Last Known Position** (Green)
   - Filled circle at last detection

3. **Predicted Modes**:
   - **Mode 1** (Blue): Highest probability
   - **Mode 2** (Orange): Second choice
   - **Mode 3** (Cyan): Alternative

4. **Future Endpoints**
   - Colored circles at predicted final positions
   - Probability labels (e.g., "65.2%")

5. **Info Panel**
   - Vehicle ID
   - History/prediction lengths
   - Mode probabilities

---

## 🔍 Use Cases

### 1. Traffic Analysis
- Predict congestion before it occurs
- Identify risky maneuvers
- Optimize traffic light timing

### 2. Autonomous Vehicles
- Anticipate other vehicles' actions
- Plan safe trajectories
- Collision avoidance

### 3. Smart Cities
- Traffic flow optimization
- Incident detection
- Infrastructure planning

### 4. Research
- Benchmark new algorithms
- Study driving behaviors
- Dataset generation

---

## 📚 References

### Papers
1. **Wayformer** (2022): Scene Transformer for autonomous driving
2. **VectorNet** (2020): Hierarchical graph neural network
3. **Social-GAN** (2018): Socially-aware trajectory prediction
4. **Trajectron++** (2020): Multi-agent trajectory forecasting

### Datasets
- **HighD**: German highway drone data
- **NGSIM**: US highway vehicle trajectories
- **INTERACTION**: Urban intersection scenarios
- **VisDrone**: Drone-based object detection

---

## 🐛 Troubleshooting

### Issue: Out of Memory

```python
# Reduce batch size
BATCH_SIZE = 16  # Instead of 32

# Reduce model size
HIDDEN_DIM = 64  # Instead of 128
NUM_LAYERS = 2   # Instead of 3
```

### Issue: Poor Predictions

1. **Check data quality**: Ensure smooth trajectories with Kalman filter
2. **Increase history**: Use more historical frames
3. **Train longer**: 50+ epochs may be needed
4. **Add more neighbors**: Increase `max_neighbors` to 10

### Issue: Slow Inference

```python
# Use GPU
predictor = TrajectoryPredictor(device='cuda')

# Reduce future length
FUTURE_LENGTH = 15  # Instead of 30

# Reduce num_modes
NUM_MODES = 1  # Single-mode prediction
```

---

## 🤝 Contributing

To extend this system:

1. **Add map context**: Integrate lane information
2. **Use better encoders**: Vision transformers for image features
3. **Multi-agent training**: Joint prediction of all vehicles
4. **Real-time optimization**: TensorRT/ONNX conversion

---

## 📝 License

See LICENSE file for details.

---

## 🎯 Next Steps

1. **Collect training data**: Process multiple drone videos
2. **Train model**: Use `train_model()` function
3. **Evaluate**: Compare against baselines (CV model, LSTM)
4. **Deploy**: Integrate with real-time traffic system

---

**For questions or issues, please open a GitHub issue.**
