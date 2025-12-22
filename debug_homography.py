import pandas as pd
import numpy as np
import json

# Load both datasets
df_yours = pd.read_csv('output/test_homography_200.csv')
df_friend = pd.read_csv('output/D2F1_lclF.csv')

# Load homography
with open('output/homography_from_d2f1.json') as f:
    H_data = json.load(f)
    H = np.array(H_data['homography_matrix'])

print('='*70)
print('  DEBUGGING HOMOGRAPHY TRANSFORMATION')
print('='*70)

# Test the homography on the correspondence points
print('\n📐 Testing Homography on Ground Control Points:')
for i, corr in enumerate(H_data['correspondences'][:3], 1):
    px, py = corr['image']
    wx_expected, wy_expected = corr['world']
    
    # Apply homography
    point_h = np.array([px, py, 1.0])
    world_h = H @ point_h
    wx_computed = world_h[0] / world_h[2]
    wy_computed = world_h[1] / world_h[2]
    
    error = np.sqrt((wx_computed - wx_expected)**2 + (wy_computed - wy_expected)**2)
    
    print(f'\nPoint {i}: ({px:.1f}, {py:.1f})px')
    print(f'  Expected: ({wx_expected:.2f}, {wy_expected:.2f})m')
    print(f'  Computed: ({wx_computed:.2f}, {wy_computed:.2f})m')
    print(f'  Error: {error:.4f}m')

# Sample random points from your data and friend's data
print('\n\n📊 Comparing Sample Points:')
print('\nYour data (first 3 points):')
for i in range(min(3, len(df_yours))):
    row = df_yours.iloc[i]
    print(f'  Frame {row["Frame"]}: ({row["X_pixel"]:.1f}, {row["Y_pixel"]:.1f})px → ({row["X_world"]:.2f}, {row["Y_world"]:.2f})m')

print('\nFriend\'s data (first 3 points):')
for i in range(3):
    row = df_friend.iloc[i]
    print(f'  Frame {row["Frame"]}: ({row["X_pixel"]:.1f}, {row["Y_pixel"]:.1f})px → ({row["X_world"]:.2f}, {row["Y_world"]:.2f})m')

# Check pixel coordinate ranges
print('\n\n📏 Pixel Coordinate Ranges:')
print(f'Your data:')
print(f'  X_pixel: {df_yours["X_pixel"].min():.1f} to {df_yours["X_pixel"].max():.1f}')
print(f'  Y_pixel: {df_yours["Y_pixel"].min():.1f} to {df_yours["Y_pixel"].max():.1f}')

print(f'Friend\'s data:')
print(f'  X_pixel: {df_friend["X_pixel"].min():.1f} to {df_friend["X_pixel"].max():.1f}')
print(f'  Y_pixel: {df_friend["Y_pixel"].min():.1f} to {df_friend["Y_pixel"].max():.1f}')

print('\n' + '='*70)
