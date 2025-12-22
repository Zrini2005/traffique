#!/usr/bin/env python3
"""
Homography Calibration Tool for Trajectory Tracker

Computes homography matrix from image coordinates to world (ground) coordinates.
Requires 4+ known correspondences (image points with known world positions).

Usage:
  python compute_homography.py --video <video_path> --output <homography.json>
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class HomographyCalibrator:
    """Interactive tool to compute homography from user-selected points"""
    
    def __init__(self, video_path: str, frame_no: int = 0):
        self.video_path = video_path
        self.frame_no = frame_no
        self.image_points = []
        self.world_points = []
        self.frame = None
        
    def load_frame(self):
        """Load frame from video"""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_no)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret or self.frame is None:
            raise ValueError(f"Cannot read frame {self.frame_no}")
        
        print(f"✓ Loaded frame {self.frame_no}")
        return self.frame
    
    def collect_points_interactive(self):
        """Collect image-world point correspondences interactively"""
        
        print("\n" + "="*70)
        print("  HOMOGRAPHY CALIBRATION - Point Collection")
        print("="*70)
        print("\nInstructions:")
        print("  1. Click on known reference points in the image")
        print("  2. Enter their world coordinates (X, Y in meters)")
        print("  3. Collect at least 4 points")
        print("  4. Press 's' to save and compute homography")
        print("  5. Press 'q' to quit\n")
        print("Good reference points:")
        print("  • Lane marking corners")
        print("  • Crosswalk edges")
        print("  • Known building corners")
        print("  • Road sign positions\n")
        
        display = self.frame.copy()
        
        def on_mouse(event, x, y, flags, param):
            nonlocal display
            if event == cv2.EVENT_LBUTTONDOWN:
                # Mark point
                cv2.circle(display, (x, y), 8, (0, 255, 0), -1)
                cv2.circle(display, (x, y), 10, (255, 255, 255), 2)
                cv2.putText(display, f"P{len(self.image_points)+1}", (x+15, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('calibrate', display)
                
                # Get world coordinates from user
                print(f"\nPoint {len(self.image_points)+1} - Image: ({x}, {y})")
                try:
                    world_x = float(input("  Enter world X (meters): "))
                    world_y = float(input("  Enter world Y (meters): "))
                    
                    self.image_points.append([x, y])
                    self.world_points.append([world_x, world_y])
                    
                    print(f"  ✓ Saved: Image({x}, {y}) → World({world_x:.2f}, {world_y:.2f})")
                    print(f"  Total points: {len(self.image_points)}")
                    
                except (ValueError, EOFError):
                    print("  ✗ Invalid input, point not added")
                    # Remove visual marker
                    display = self.frame.copy()
                    for i, (px, py) in enumerate(self.image_points):
                        cv2.circle(display, (px, py), 8, (0, 255, 0), -1)
                        cv2.circle(display, (px, py), 10, (255, 255, 255), 2)
                        cv2.putText(display, f"P{i+1}", (px+15, py-15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('calibrate', display)
        
        cv2.namedWindow('calibrate', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('calibrate', 1280, 720)
        cv2.imshow('calibrate', display)
        cv2.setMouseCallback('calibrate', on_mouse)
        
        while True:
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('s'):
                if len(self.image_points) >= 4:
                    print(f"\n✓ Saving {len(self.image_points)} points")
                    break
                else:
                    print(f"\n⚠️  Need at least 4 points (have {len(self.image_points)})")
            
            elif key == ord('q'):
                print("\n⚠️  Calibration cancelled")
                cv2.destroyAllWindows()
                return False
            
            elif key == ord('u'):  # Undo last point
                if self.image_points:
                    self.image_points.pop()
                    self.world_points.pop()
                    display = self.frame.copy()
                    for i, (px, py) in enumerate(self.image_points):
                        cv2.circle(display, (px, py), 8, (0, 255, 0), -1)
                        cv2.circle(display, (px, py), 10, (255, 255, 255), 2)
                        cv2.putText(display, f"P{i+1}", (px+15, py-15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow('calibrate', display)
                    print(f"  ↩️  Undid last point. Total: {len(self.image_points)}")
        
        cv2.destroyAllWindows()
        return True
    
    def compute_homography(self):
        """Compute homography matrix from collected points"""
        
        if len(self.image_points) < 4:
            raise ValueError("Need at least 4 points to compute homography")
        
        src_pts = np.array(self.image_points, dtype=np.float32)
        dst_pts = np.array(self.world_points, dtype=np.float32)
        
        # Compute homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            raise ValueError("Failed to compute homography")
        
        # Compute reprojection error
        errors = []
        for img_pt, world_pt in zip(src_pts, dst_pts):
            # Transform image point to world
            img_pt_h = np.array([img_pt[0], img_pt[1], 1.0])
            world_pt_pred = H @ img_pt_h
            world_pt_pred = world_pt_pred[:2] / world_pt_pred[2]
            
            error = np.linalg.norm(world_pt_pred - world_pt)
            errors.append(error)
        
        print(f"\n📐 Homography computed!")
        print(f"   Reprojection error: {np.mean(errors):.4f} ± {np.std(errors):.4f} meters")
        print(f"   Max error: {np.max(errors):.4f} meters")
        
        return H
    
    def save_homography(self, H: np.ndarray, output_path: str):
        """Save homography to JSON"""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "homography_matrix": H.tolist(),
            "source": "image_pixels",
            "target": "world_meters",
            "points_used": len(self.image_points),
            "correspondences": [
                {
                    "image": [float(img[0]), float(img[1])],
                    "world": [float(world[0]), float(world[1])]
                }
                for img, world in zip(self.image_points, self.world_points)
            ],
            "video": self.video_path,
            "frame": self.frame_no
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Saved homography to: {output_file}")
        
        # Print usage instructions
        print(f"\n📝 Usage:")
        print(f"   python trajectory_tracker.py <video> --homography \"{output_file}\" ...")
    
    def run(self, output_path: str):
        """Run full calibration workflow"""
        
        self.load_frame()
        success = self.collect_points_interactive()
        
        if not success:
            return
        
        H = self.compute_homography()
        self.save_homography(H, output_path)


def load_homography_from_file(json_path: str) -> np.ndarray:
    """Load homography matrix from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    H = np.array(data['homography_matrix'], dtype=np.float32)
    print(f"✓ Loaded homography from {json_path}")
    print(f"  Points used: {data.get('points_used', 'unknown')}")
    
    return H


def transform_point(point: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Transform a point from image to world coordinates using homography
    
    Args:
        point: [x, y] in image pixels
        H: 3x3 homography matrix
    
    Returns:
        [x, y] in world meters
    """
    point_h = np.array([point[0], point[1], 1.0])
    world_h = H @ point_h
    world = world_h[:2] / world_h[2]
    return world


def main():
    parser = argparse.ArgumentParser(
        description="Compute homography for trajectory tracker"
    )
    parser.add_argument("--video", required=True, help="Video file path")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to use for calibration")
    parser.add_argument("--output", default="output/homography.json", help="Output JSON file")
    
    args = parser.parse_args()
    
    calibrator = HomographyCalibrator(args.video, args.frame)
    calibrator.run(args.output)


if __name__ == "__main__":
    main()
