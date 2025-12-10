#!/usr/bin/env python3
"""
Trajectory Prediction Evaluation Metrics
Implements standard metrics: ADE, FDE, MissRate, etc.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class PredictionMetrics:
    """Container for prediction evaluation metrics"""
    ade: float  # Average Displacement Error (meters)
    fde: float  # Final Displacement Error (meters)
    miss_rate: float  # Percentage of predictions > 2m from ground truth
    mode_selection_accuracy: float  # Accuracy of selecting best mode
    min_ade: float  # ADE of best mode
    min_fde: float  # FDE of best mode


class TrajectoryEvaluator:
    """
    Evaluate trajectory prediction performance
    Implements standard metrics from trajectory prediction literature
    """
    
    def __init__(self, miss_threshold: float = 2.0):
        """
        Args:
            miss_threshold: Distance threshold (meters) for miss rate
        """
        self.miss_threshold = miss_threshold
    
    def compute_ade(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Average Displacement Error
        Average L2 distance between predicted and ground truth across all time steps
        
        Args:
            predicted: (T, 2) predicted trajectory
            ground_truth: (T, 2) ground truth trajectory
        
        Returns:
            ADE in meters
        """
        distances = np.linalg.norm(predicted - ground_truth, axis=1)
        return np.mean(distances)
    
    def compute_fde(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Final Displacement Error
        L2 distance between final predicted and ground truth positions
        
        Args:
            predicted: (T, 2) predicted trajectory
            ground_truth: (T, 2) ground truth trajectory
        
        Returns:
            FDE in meters
        """
        return np.linalg.norm(predicted[-1] - ground_truth[-1])
    
    def compute_miss_rate(
        self,
        predicted: np.ndarray,
        ground_truth: np.ndarray
    ) -> bool:
        """
        Check if prediction is a "miss"
        A prediction is a miss if FDE > threshold
        
        Args:
            predicted: (T, 2) predicted trajectory
            ground_truth: (T, 2) ground truth trajectory
        
        Returns:
            True if miss, False if hit
        """
        fde = self.compute_fde(predicted, ground_truth)
        return fde > self.miss_threshold
    
    def compute_multimodal_metrics(
        self,
        predicted_modes: np.ndarray,
        mode_probabilities: np.ndarray,
        ground_truth: np.ndarray
    ) -> PredictionMetrics:
        """
        Compute metrics for multi-modal prediction
        
        Args:
            predicted_modes: (num_modes, T, 2) predicted trajectories
            mode_probabilities: (num_modes,) probability of each mode
            ground_truth: (T, 2) ground truth trajectory
        
        Returns:
            PredictionMetrics object
        """
        num_modes = len(predicted_modes)
        
        # Compute metrics for each mode
        ades = []
        fdes = []
        
        for mode_traj in predicted_modes:
            ade = self.compute_ade(mode_traj, ground_truth)
            fde = self.compute_fde(mode_traj, ground_truth)
            ades.append(ade)
            fdes.append(fde)
        
        ades = np.array(ades)
        fdes = np.array(fdes)
        
        # Best mode (minimum ADE)
        best_mode_idx = np.argmin(ades)
        min_ade = ades[best_mode_idx]
        min_fde = fdes[best_mode_idx]
        
        # Mode selection accuracy
        # Did we assign highest probability to the best mode?
        predicted_mode = np.argmax(mode_probabilities)
        mode_selection_accuracy = 1.0 if predicted_mode == best_mode_idx else 0.0
        
        # Weighted ADE/FDE by mode probabilities
        weighted_ade = np.sum(ades * mode_probabilities)
        weighted_fde = np.sum(fdes * mode_probabilities)
        
        # Miss rate
        miss = fdes[predicted_mode] > self.miss_threshold
        miss_rate = 1.0 if miss else 0.0
        
        return PredictionMetrics(
            ade=weighted_ade,
            fde=weighted_fde,
            miss_rate=miss_rate,
            mode_selection_accuracy=mode_selection_accuracy,
            min_ade=min_ade,
            min_fde=min_fde
        )
    
    def evaluate_dataset(
        self,
        predictions: List[Tuple[np.ndarray, np.ndarray]],
        ground_truths: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate entire dataset
        
        Args:
            predictions: List of (predicted_modes, mode_probs) tuples
            ground_truths: List of ground truth trajectories
        
        Returns:
            Dictionary of aggregated metrics
        """
        all_metrics = []
        
        for (pred_modes, mode_probs), gt in zip(predictions, ground_truths):
            metrics = self.compute_multimodal_metrics(pred_modes, mode_probs, gt)
            all_metrics.append(metrics)
        
        # Aggregate
        avg_ade = np.mean([m.ade for m in all_metrics])
        avg_fde = np.mean([m.fde for m in all_metrics])
        avg_miss_rate = np.mean([m.miss_rate for m in all_metrics])
        avg_mode_accuracy = np.mean([m.mode_selection_accuracy for m in all_metrics])
        avg_min_ade = np.mean([m.min_ade for m in all_metrics])
        avg_min_fde = np.mean([m.min_fde for m in all_metrics])
        
        return {
            'ADE': avg_ade,
            'FDE': avg_fde,
            'minADE': avg_min_ade,
            'minFDE': avg_min_fde,
            'MissRate': avg_miss_rate * 100,  # Percentage
            'ModeAccuracy': avg_mode_accuracy * 100
        }


def pixel_to_meters(positions: np.ndarray, pixels_per_meter: float = 20.0) -> np.ndarray:
    """
    Convert pixel coordinates to meters
    
    Args:
        positions: (N, 2) positions in pixels
        pixels_per_meter: Calibration factor
    
    Returns:
        positions in meters
    """
    return positions / pixels_per_meter


def meters_to_pixels(positions: np.ndarray, pixels_per_meter: float = 20.0) -> np.ndarray:
    """
    Convert meter coordinates to pixels
    
    Args:
        positions: (N, 2) positions in meters
        pixels_per_meter: Calibration factor
    
    Returns:
        positions in pixels
    """
    return positions * pixels_per_meter


class DatasetBuilder:
    """
    Build training/validation datasets from trajectory data
    Handles data splitting, augmentation, and normalization
    """
    
    def __init__(
        self,
        history_length: int = 20,
        future_length: int = 30,
        min_trajectory_length: int = 60
    ):
        self.history_length = history_length
        self.future_length = future_length
        self.min_trajectory_length = min_trajectory_length
    
    def build_dataset_from_trajectories(
        self,
        trajectories: Dict[int, List[np.ndarray]],
        pixels_per_meter: float = 20.0
    ) -> List[Dict]:
        """
        Build dataset from raw trajectories
        
        Args:
            trajectories: Dict mapping vehicle_id to list of positions
            pixels_per_meter: Conversion factor
        
        Returns:
            List of training samples
        """
        dataset = []
        
        for vehicle_id, positions in trajectories.items():
            positions = np.array(positions)
            
            if len(positions) < self.min_trajectory_length:
                continue
            
            # Convert to meters
            positions = pixel_to_meters(positions, pixels_per_meter)
            
            # Create sliding windows
            for i in range(len(positions) - self.history_length - self.future_length):
                history = positions[i:i + self.history_length]
                future = positions[i + self.history_length:i + self.history_length + self.future_length]
                
                # Normalize: relative to last historical position
                last_pos = history[-1]
                history_norm = history - last_pos
                future_norm = future - last_pos
                
                # Compute velocities and accelerations
                velocities = np.gradient(history_norm, axis=0)
                accelerations = np.gradient(velocities, axis=0)
                
                # Combine features
                ego_trajectory = np.concatenate([
                    history_norm,
                    velocities,
                    accelerations
                ], axis=-1)
                
                # For now, use dummy neighbors (would need actual neighbor data)
                neighbor_trajectories = np.zeros((1, self.history_length, 6))
                
                sample = {
                    'ego_trajectory': ego_trajectory.astype(np.float32),
                    'neighbor_trajectories': neighbor_trajectories.astype(np.float32),
                    'future_trajectory': future_norm.astype(np.float32),
                    'vehicle_id': vehicle_id
                }
                
                dataset.append(sample)
        
        return dataset
    
    def train_val_split(
        self,
        dataset: List[Dict],
        val_ratio: float = 0.2,
        random_seed: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split dataset into train and validation sets
        
        Args:
            dataset: Full dataset
            val_ratio: Fraction for validation
            random_seed: Random seed for reproducibility
        
        Returns:
            train_data, val_data
        """
        np.random.seed(random_seed)
        
        indices = np.random.permutation(len(dataset))
        split_idx = int(len(dataset) * (1 - val_ratio))
        
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        
        train_data = [dataset[i] for i in train_indices]
        val_data = [dataset[i] for i in val_indices]
        
        return train_data, val_data


if __name__ == "__main__":
    # Example usage
    print("Trajectory Prediction Evaluation")
    print("="*70)
    
    evaluator = TrajectoryEvaluator(miss_threshold=2.0)
    
    # Example predictions
    predicted_modes = np.array([
        [[0, 0], [1, 1], [2, 2], [3, 3]],  # Mode 1
        [[0, 0], [1, 0.5], [2, 1], [3, 1.5]],  # Mode 2
        [[0, 0], [0.5, 1], [1, 2], [1.5, 3]]  # Mode 3
    ])
    
    mode_probs = np.array([0.6, 0.3, 0.1])
    ground_truth = np.array([[0, 0], [1, 0.8], [2, 1.6], [3, 2.4]])
    
    metrics = evaluator.compute_multimodal_metrics(
        predicted_modes,
        mode_probs,
        ground_truth
    )
    
    print(f"\nSample Evaluation:")
    print(f"  ADE: {metrics.ade:.3f} meters")
    print(f"  FDE: {metrics.fde:.3f} meters")
    print(f"  Min ADE: {metrics.min_ade:.3f} meters")
    print(f"  Min FDE: {metrics.min_fde:.3f} meters")
    print(f"  Miss Rate: {metrics.miss_rate*100:.1f}%")
    print(f"  Mode Selection Accuracy: {metrics.mode_selection_accuracy*100:.1f}%")
