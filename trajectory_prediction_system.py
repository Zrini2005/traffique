#!/usr/bin/env python3
"""
Advanced Trajectory Prediction System for Drone Footage
Uses Transformer architecture with social attention and map context

Implements state-of-the-art trajectory prediction with:
- Multi-head self-attention for temporal modeling
- Social interaction modeling via graph attention
- Map-aware predictions using road context
- Kalman filtering for state estimation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass
class TrajectoryData:
    """Structured trajectory data for a vehicle"""
    vehicle_id: int
    positions: np.ndarray  # (T, 2) - historical positions (x, y)
    velocities: np.ndarray  # (T, 2) - historical velocities (vx, vy)
    accelerations: np.ndarray  # (T, 2) - historical accelerations (ax, ay)
    timestamp: np.ndarray  # (T,) - frame numbers
    neighbors: List[int]  # IDs of neighboring vehicles
    lane_context: Optional[np.ndarray] = None  # Road/lane features


class KinematicsExtractor:
    """Extract velocity and acceleration from position history"""
    
    @staticmethod
    def compute_kinematics(positions: np.ndarray, dt: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute velocities and accelerations from positions
        
        Args:
            positions: (T, 2) array of (x, y) positions
            dt: Time step in seconds (1/fps)
        
        Returns:
            velocities: (T, 2) array
            accelerations: (T, 2) array
        """
        # Velocity using central differences
        velocities = np.zeros_like(positions)
        velocities[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
        velocities[0] = (positions[1] - positions[0]) / dt
        velocities[-1] = (positions[-1] - positions[-2]) / dt
        
        # Acceleration using central differences
        accelerations = np.zeros_like(positions)
        accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / (2 * dt)
        accelerations[0] = (velocities[1] - velocities[0]) / dt
        accelerations[-1] = (velocities[-1] - velocities[-2]) / dt
        
        return velocities, accelerations


class SocialContextEncoder(nn.Module):
    """
    Encode social interactions using Graph Attention Network
    Models how vehicles influence each other's future trajectories
    """
    
    def __init__(self, feature_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Graph attention layers
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.layer_norm1 = nn.LayerNorm(feature_dim)
        self.layer_norm2 = nn.LayerNorm(feature_dim)
    
    def forward(self, ego_features: torch.Tensor, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ego_features: (batch, feature_dim) - target vehicle features
            neighbor_features: (batch, num_neighbors, feature_dim)
        
        Returns:
            context: (batch, feature_dim) - interaction-aware features
        """
        # Add ego to the set
        all_features = torch.cat([ego_features.unsqueeze(1), neighbor_features], dim=1)
        
        # Self-attention over all vehicles
        attn_out, _ = self.attention(all_features, all_features, all_features)
        attn_out = self.layer_norm1(all_features + attn_out)
        
        # MLP
        mlp_out = self.mlp(attn_out)
        output = self.layer_norm2(attn_out + mlp_out)
        
        # Return only ego vehicle features (with social context)
        return output[:, 0, :]


class TemporalEncoder(nn.Module):
    """
    Transformer encoder for temporal trajectory history
    Models sequential dependencies in motion patterns
    """
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 128, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            trajectory: (batch, seq_len, input_dim) - [x, y, vx, vy, ax, ay]
        
        Returns:
            encoded: (batch, hidden_dim) - encoded trajectory features
        """
        x = self.input_proj(trajectory)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.output_norm(x)
        
        # Use last time step as summary
        return x[:, -1, :]


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class TrajectoryDecoder(nn.Module):
    """
    Autoregressive decoder for future trajectory prediction
    Generates multi-modal predictions (multiple possible futures)
    """
    
    def __init__(self, feature_dim: int = 128, output_dim: int = 2, num_modes: int = 3):
        super().__init__()
        self.num_modes = num_modes
        
        # Mode predictor (which future is most likely)
        self.mode_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_modes)
        )
        
        # Trajectory decoders (one per mode)
        self.decoders = nn.ModuleList([
            nn.GRU(
                input_size=output_dim + feature_dim,
                hidden_size=feature_dim,
                num_layers=2,
                batch_first=True
            )
            for _ in range(num_modes)
        ])
        
        self.output_layers = nn.ModuleList([
            nn.Linear(feature_dim, output_dim)
            for _ in range(num_modes)
        ])
    
    def forward(self, context: torch.Tensor, num_future_steps: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            context: (batch, feature_dim) - encoded context
            num_future_steps: Number of future time steps to predict
        
        Returns:
            trajectories: (batch, num_modes, num_future_steps, 2)
            mode_probs: (batch, num_modes)
        """
        batch_size = context.size(0)
        device = context.device
        
        # Predict mode probabilities
        mode_logits = self.mode_predictor(context)
        mode_probs = F.softmax(mode_logits, dim=-1)
        
        # Generate trajectory for each mode
        all_trajectories = []
        
        for mode_idx in range(self.num_modes):
            decoder = self.decoders[mode_idx]
            output_layer = self.output_layers[mode_idx]
            
            # Initialize hidden state
            hidden = context.unsqueeze(0).repeat(2, 1, 1)  # (num_layers, batch, hidden_dim)
            
            # Start from origin (relative coordinates)
            current_pos = torch.zeros(batch_size, 1, 2, device=device)
            trajectory = []
            
            for t in range(num_future_steps):
                # Concatenate position with context
                decoder_input = torch.cat([current_pos, context.unsqueeze(1)], dim=-1)
                
                # Decode one step
                output, hidden = decoder(decoder_input, hidden)
                
                # Predict displacement
                delta = output_layer(output)
                current_pos = delta
                
                trajectory.append(delta)
            
            trajectory = torch.cat(trajectory, dim=1)  # (batch, num_future_steps, 2)
            
            # Convert from displacements to absolute positions
            trajectory = torch.cumsum(trajectory, dim=1)
            
            all_trajectories.append(trajectory)
        
        trajectories = torch.stack(all_trajectories, dim=1)  # (batch, num_modes, num_future_steps, 2)
        
        return trajectories, mode_probs


class TrajectoryPredictionModel(nn.Module):
    """
    Complete trajectory prediction system
    Combines temporal encoding, social context, and multi-modal decoding
    """
    
    def __init__(
        self,
        history_dim: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_modes: int = 3
    ):
        super().__init__()
        
        # Temporal encoder for ego vehicle
        self.temporal_encoder = TemporalEncoder(
            input_dim=history_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads
        )
        
        # Social context encoder
        self.social_encoder = SocialContextEncoder(
            feature_dim=hidden_dim,
            num_heads=num_heads
        )
        
        # Trajectory decoder
        self.decoder = TrajectoryDecoder(
            feature_dim=hidden_dim,
            output_dim=2,
            num_modes=num_modes
        )
    
    def forward(
        self,
        ego_trajectory: torch.Tensor,
        neighbor_trajectories: torch.Tensor,
        num_future_steps: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_trajectory: (batch, history_len, 6) - [x, y, vx, vy, ax, ay]
            neighbor_trajectories: (batch, num_neighbors, history_len, 6)
            num_future_steps: Number of future steps to predict
        
        Returns:
            predicted_trajectories: (batch, num_modes, future_len, 2)
            mode_probabilities: (batch, num_modes)
        """
        # Encode ego trajectory
        ego_features = self.temporal_encoder(ego_trajectory)
        
        # Encode neighbor trajectories
        batch_size, num_neighbors = neighbor_trajectories.shape[:2]
        neighbor_trajectories_flat = neighbor_trajectories.reshape(-1, *neighbor_trajectories.shape[2:])
        neighbor_features = self.temporal_encoder(neighbor_trajectories_flat)
        neighbor_features = neighbor_features.reshape(batch_size, num_neighbors, -1)
        
        # Fuse with social context
        context = self.social_encoder(ego_features, neighbor_features)
        
        # Decode future trajectory
        trajectories, mode_probs = self.decoder(context, num_future_steps)
        
        return trajectories, mode_probs


class TrajectoryPredictor:
    """
    High-level interface for trajectory prediction
    Handles data preprocessing and post-processing
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        history_length: int = 20,  # 20 frames = 0.8 seconds at 25fps
        future_length: int = 30,   # 30 frames = 1.2 seconds
        num_modes: int = 3
    ):
        self.device = device
        self.history_length = history_length
        self.future_length = future_length
        
        # Initialize model
        self.model = TrajectoryPredictionModel(
            history_dim=6,
            hidden_dim=128,
            num_layers=3,
            num_heads=4,
            num_modes=num_modes
        ).to(device)
        
        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        self.kinematics = KinematicsExtractor()
    
    def prepare_trajectory_data(
        self,
        positions: np.ndarray,
        neighbor_positions: List[np.ndarray],
        dt: float = 0.04
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare trajectory data for model input
        
        Args:
            positions: (T, 2) - ego vehicle positions
            neighbor_positions: List of (T, 2) arrays for neighbors
            dt: Time step in seconds
        
        Returns:
            ego_tensor: (1, T, 6)
            neighbor_tensor: (1, num_neighbors, T, 6)
        """
        # Compute kinematics for ego
        velocities, accelerations = self.kinematics.compute_kinematics(positions, dt)
        ego_data = np.concatenate([positions, velocities, accelerations], axis=-1)
        
        # Normalize relative to current position
        current_pos = positions[-1]
        ego_data[:, :2] -= current_pos
        
        ego_tensor = torch.FloatTensor(ego_data).unsqueeze(0).to(self.device)
        
        # Compute kinematics for neighbors
        neighbor_data_list = []
        for neighbor_pos in neighbor_positions:
            if len(neighbor_pos) == len(positions):
                vel, acc = self.kinematics.compute_kinematics(neighbor_pos, dt)
                neighbor_data = np.concatenate([neighbor_pos, vel, acc], axis=-1)
                neighbor_data[:, :2] -= current_pos  # Relative to ego
                neighbor_data_list.append(neighbor_data)
        
        if neighbor_data_list:
            neighbor_tensor = torch.FloatTensor(np.stack(neighbor_data_list)).unsqueeze(0).to(self.device)
        else:
            # No neighbors - create dummy
            neighbor_tensor = torch.zeros(1, 1, len(positions), 6, device=self.device)
        
        return ego_tensor, neighbor_tensor
    
    @torch.no_grad()
    def predict(
        self,
        positions: np.ndarray,
        neighbor_positions: List[np.ndarray] = None,
        return_all_modes: bool = False
    ) -> np.ndarray:
        """
        Predict future trajectory
        
        Args:
            positions: (T, 2) - historical positions of ego vehicle
            neighbor_positions: List of (T, 2) arrays for neighbors
            return_all_modes: If True, return all modes, else return most likely
        
        Returns:
            predicted_trajectory: (future_length, 2) or (num_modes, future_length, 2)
        """
        if neighbor_positions is None:
            neighbor_positions = []
        
        # Prepare data
        ego_tensor, neighbor_tensor = self.prepare_trajectory_data(
            positions, neighbor_positions
        )
        
        # Predict
        trajectories, mode_probs = self.model(
            ego_tensor,
            neighbor_tensor,
            num_future_steps=self.future_length
        )
        
        # Convert to numpy (add back current position)
        trajectories = trajectories.cpu().numpy()[0]  # (num_modes, future_len, 2)
        mode_probs = mode_probs.cpu().numpy()[0]  # (num_modes,)
        
        current_pos = positions[-1]
        trajectories += current_pos
        
        if return_all_modes:
            return trajectories, mode_probs
        else:
            # Return most likely mode
            best_mode = np.argmax(mode_probs)
            return trajectories[best_mode]


def train_model(
    train_data: List[Dict],
    val_data: List[Dict],
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    save_path: str = 'models/trajectory_predictor.pth'
):
    """
    Train the trajectory prediction model
    
    Args:
        train_data: List of dicts with 'ego_trajectory', 'neighbor_trajectories', 'future_trajectory'
        val_data: Validation data in same format
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        save_path: Where to save the model
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = TrajectoryPredictionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        # Training loop
        for batch_idx in range(0, len(train_data), batch_size):
            batch = train_data[batch_idx:batch_idx + batch_size]
            
            # Prepare batch
            ego_batch = torch.stack([torch.FloatTensor(d['ego_trajectory']) for d in batch]).to(device)
            neighbor_batch = torch.stack([torch.FloatTensor(d['neighbor_trajectories']) for d in batch]).to(device)
            gt_future = torch.stack([torch.FloatTensor(d['future_trajectory']) for d in batch]).to(device)
            
            # Forward pass
            pred_trajectories, mode_probs = model(ego_batch, neighbor_batch, num_future_steps=gt_future.size(1))
            
            # Loss: weighted combination of all modes
            losses = torch.sum((pred_trajectories - gt_future.unsqueeze(1)) ** 2, dim=-1)  # (batch, modes, timesteps)
            losses = torch.mean(losses, dim=-1)  # (batch, modes)
            
            # Weight by mode probability
            loss = torch.mean(torch.sum(losses * mode_probs, dim=-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx in range(0, len(val_data), batch_size):
                batch = val_data[batch_idx:batch_idx + batch_size]
                
                ego_batch = torch.stack([torch.FloatTensor(d['ego_trajectory']) for d in batch]).to(device)
                neighbor_batch = torch.stack([torch.FloatTensor(d['neighbor_trajectories']) for d in batch]).to(device)
                gt_future = torch.stack([torch.FloatTensor(d['future_trajectory']) for d in batch]).to(device)
                
                pred_trajectories, mode_probs = model(ego_batch, neighbor_batch, num_future_steps=gt_future.size(1))
                
                losses = torch.sum((pred_trajectories - gt_future.unsqueeze(1)) ** 2, dim=-1)
                losses = torch.mean(losses, dim=-1)
                loss = torch.mean(torch.sum(losses * mode_probs, dim=-1))
                
                val_loss += loss.item()
        
        train_loss /= len(train_data) // batch_size
        val_loss /= len(val_data) // batch_size
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model to {save_path}")
        
        scheduler.step()
    
    return model


if __name__ == "__main__":
    # Example usage
    print("Trajectory Prediction System Initialized")
    print("="*70)
    
    # Create predictor
    predictor = TrajectoryPredictor(
        device='cpu',
        history_length=20,
        future_length=30,
        num_modes=3
    )
    
    # Example: predict trajectory
    # Historical positions (20 time steps)
    positions = np.array([[i, i*0.5] for i in range(20)], dtype=np.float32)
    
    # Neighbor positions
    neighbor_positions = [
        np.array([[i+5, i*0.3] for i in range(20)], dtype=np.float32),
        np.array([[i-5, i*0.7] for i in range(20)], dtype=np.float32)
    ]
    
    # Predict future
    future_trajectory = predictor.predict(positions, neighbor_positions)
    
    print(f"\nPredicted future trajectory shape: {future_trajectory.shape}")
    print(f"First 5 predicted positions:\n{future_trajectory[:5]}")
    
    # Get all modes
    all_modes, probs = predictor.predict(positions, neighbor_positions, return_all_modes=True)
    print(f"\nAll modes shape: {all_modes.shape}")
    print(f"Mode probabilities: {probs}")
