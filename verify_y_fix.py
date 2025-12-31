#!/usr/bin/env python3
"""Verify Y-axis fix by comparing before/after"""

import pandas as pd
import numpy as np

# Load the new trajectories
csv_new = pd.read_csv('/mnt/c/Users/srini/Downloads/trajectories_new_200frames.csv')
csv_gt = pd.read_csv('/mnt/c/Users/srini/Downloads/D2F1_lclF (1).csv')

# Rename columns
csv_new.rename(columns={'VehicleID': 'vid', 'Frame': 'frame', 'X_pixel': 'x', 'Y_pixel': 'y'}, inplace=True)
csv_gt.rename(columns={'VehicleID': 'vid', 'Frame': 'frame', 'X_pixel': 'x', 'Y_pixel': 'y'}, inplace=True)

# Analyze motor_77 vs Bike_5 (best match)
df1 = csv_new[csv_new['vid'] == 'motor_77'].sort_values('frame')
df2 = csv_gt[csv_gt['vid'] == 'Bike_5'].sort_values('frame')

common_frames = sorted(set(df1['frame']) & set(df2['frame']))
df1 = df1[df1['frame'].isin(common_frames)].set_index('frame').sort_index()
df2 = df2[df2['frame'].isin(common_frames)].set_index('frame').sort_index()

# Calculate frame-to-frame movements
y1_diff = np.abs(np.diff(df1['y'].values))
y2_diff = np.abs(np.diff(df2['y'].values))

x1_diff = np.abs(np.diff(df1['x'].values))
x2_diff = np.abs(np.diff(df2['x'].values))

# Calculate errors
y_errors = np.abs(df1['y'].values - df2['y'].values)
x_errors = np.abs(df1['x'].values - df2['x'].values)

print("="*70)
print("Y-AXIS FIX VERIFICATION (motor_77 vs Bike_5)")
print("="*70)

print("\n[X-AXIS - Should remain UNCHANGED]")
print(f"  Mean movement/frame: {x1_diff.mean():.3f} px (GT: {x2_diff.mean():.3f})")
print(f"  Max jump: {x1_diff.max():.3f} px (GT: {x2_diff.max():.3f})")
print(f"  Error: {x_errors.mean():.3f} ± {x_errors.std():.3f} px")

print("\n[Y-AXIS - Should be IMPROVED]")
print(f"  Mean movement/frame: {y1_diff.mean():.3f} px (GT: {y2_diff.mean():.3f})")
print(f"  Max jump: {y1_diff.max():.3f} px (GT: {y2_diff.max():.3f})")
print(f"  Error: {y_errors.mean():.3f} ± {y_errors.std():.3f} px")
print(f"  Movement ratio (yours/GT): {y1_diff.mean() / y2_diff.mean():.2f}x")

print("\n[OVERALL METRICS]")
print(f"  Y/X error ratio: {y_errors.mean() / x_errors.mean():.2f}x")
print(f"  Y std: {df1['y'].std():.3f} px (GT: {df2['y'].std():.3f})")

# Status
if y1_diff.mean() < 0.15:
    print("\n✅ SUCCESS: Y movement reduced to near ground truth levels!")
elif y1_diff.mean() < 0.20:
    print("\n⚠️  PARTIAL: Y movement improved but can be better")
else:
    print("\n❌ FAILED: Y movement still too high")

if y_errors.mean() <= x_errors.mean() * 1.2:
    print("✅ SUCCESS: Y error now comparable to X error!")
