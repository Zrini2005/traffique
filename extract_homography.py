#!/usr/bin/env python3
"""
Extract specific homography matrix from camera_calibration.json

Usage:
  python extract_homography.py camera_calibration.json D1_to_D2 --output output/homography_d1_to_d2.json
"""

import json
import argparse
import numpy as np
from pathlib import Path


def extract_homography(calibration_file: str, matrix_name: str, output_file: str):
    """
    Extract a specific homography matrix from calibration JSON
    
    Args:
        calibration_file: Path to camera_calibration.json
        matrix_name: Name of matrix (e.g., "D1_to_D2", "D2_to_D3")
        output_file: Output JSON file path
    """
    
    # Load calibration file
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    
    # Find the matrix
    if matrix_name not in calibration_data:
        available = [k for k in calibration_data.keys() if '_to_' in k]
        raise ValueError(
            f"Matrix '{matrix_name}' not found in calibration file.\n"
            f"Available matrices: {', '.join(available)}"
        )
    
    matrix_data = calibration_data[matrix_name]
    
    # Extract homography matrix
    if isinstance(matrix_data, dict) and 'homography' in matrix_data:
        H = matrix_data['homography']
    elif isinstance(matrix_data, list):
        H = matrix_data
    else:
        raise ValueError(f"Unexpected format for matrix '{matrix_name}'")
    
    # Convert to numpy array and validate
    H = np.array(H, dtype=np.float32)
    
    if H.shape != (3, 3):
        raise ValueError(f"Invalid homography shape: {H.shape}, expected (3, 3)")
    
    # Create output data
    output_data = {
        "homography_matrix": H.tolist(),
        "source": "image_pixels",
        "target": "world_meters",
        "matrix_name": matrix_name,
        "extracted_from": calibration_file
    }
    
    # Add metadata if available
    if isinstance(matrix_data, dict):
        if 'points_used' in matrix_data:
            output_data['points_used'] = matrix_data['points_used']
    
    # Save to output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Extracted homography matrix: {matrix_name}")
    print(f"   Saved to: {output_path}")
    print(f"\n📐 Homography Matrix:")
    print(H)
    
    # Show usage
    print(f"\n📝 Usage:")
    print(f'   python trajectory_tracker.py <video> --homography "{output_file}" ...')


def list_available_matrices(calibration_file: str):
    """List all available homography matrices in calibration file"""
    
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    
    matrices = [k for k in calibration_data.keys() if '_to_' in k]
    
    print(f"\n📋 Available Homography Matrices in {calibration_file}:")
    print("─" * 70)
    
    for matrix_name in matrices:
        matrix_data = calibration_data[matrix_name]
        
        # Get matrix
        if isinstance(matrix_data, dict) and 'homography' in matrix_data:
            H = np.array(matrix_data['homography'])
            points = matrix_data.get('points_used', 'unknown')
        elif isinstance(matrix_data, list):
            H = np.array(matrix_data)
            points = 'unknown'
        else:
            continue
        
        print(f"  • {matrix_name}")
        print(f"    Shape: {H.shape}, Points used: {points}")
    
    print("─" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Extract homography matrix from camera_calibration.json"
    )
    parser.add_argument("calibration_file", help="Path to camera_calibration.json")
    parser.add_argument("matrix_name", nargs='?', help="Matrix name (e.g., D1_to_D2)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--list", action="store_true", help="List available matrices")
    
    args = parser.parse_args()
    
    if args.list or not args.matrix_name:
        list_available_matrices(args.calibration_file)
        if not args.matrix_name:
            return
    
    if not args.output:
        args.output = f"output/homography_{args.matrix_name.lower()}.json"
    
    extract_homography(args.calibration_file, args.matrix_name, args.output)


if __name__ == "__main__":
    main()
