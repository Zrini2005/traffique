import pandas as pd
import numpy as np

df = pd.read_csv('output/D2F1_lclF.csv')

print('Columns:', list(df.columns))
print('\n=== Coordinate Analysis ===')
print(f'X_pixel: {df["X_pixel"].min():.1f} to {df["X_pixel"].max():.1f}')
print(f'Y_pixel: {df["Y_pixel"].min():.1f} to {df["Y_pixel"].max():.1f}')
print(f'X_world: {df["X_world"].min():.1f} to {df["X_world"].max():.1f}')
print(f'Y_world: {df["Y_world"].min():.1f} to {df["Y_world"].max():.1f}')

print('\n=== Sample data ===')
print(df[['X_pixel', 'Y_pixel', 'X_world', 'Y_world']].head(10))

print('\n=== Checking if simple scale or homography ===')
df['scale_x'] = df['X_world'] / df['X_pixel']
df['scale_y'] = df['Y_world'] / df['Y_pixel']

print(f'X scale factor: mean={df["scale_x"].mean():.6f}, std={df["scale_x"].std():.6f}')
print(f'Y scale factor: mean={df["scale_y"].mean():.6f}, std={df["scale_y"].std():.6f}')

print(f'\nScale variation (Coefficient of Variation):')
print(f'  X: {df["scale_x"].std()/df["scale_x"].mean()*100:.2f}%')
print(f'  Y: {df["scale_y"].std()/df["scale_y"].mean()*100:.2f}%')

print('\n=== Scale variation across spatial positions ===')
# Check if scale varies by position (indicates homography)
bins = 5
df['x_bin'] = pd.cut(df['X_pixel'], bins=bins, labels=False)
df['y_bin'] = pd.cut(df['Y_pixel'], bins=bins, labels=False)

print('\nScale by horizontal position (left to right):')
for i in range(bins):
    subset = df[df['x_bin'] == i]
    if len(subset) > 0:
        print(f'  Bin {i}: scale_x={subset["scale_x"].mean():.6f}')

print('\nScale by vertical position (top to bottom):')
for i in range(bins):
    subset = df[df['y_bin'] == i]
    if len(subset) > 0:
        print(f'  Bin {i}: scale_y={subset["scale_y"].mean():.6f}')

# Test if it's truly homography
print('\n=== Homography Test ===')
scale_x_var = df['scale_x'].std() / df['scale_x'].mean()
scale_y_var = df['scale_y'].std() / df['scale_y'].mean()

if scale_x_var < 0.01 and scale_y_var < 0.01:
    print('✓ SIMPLE LINEAR SCALE (constant scale factors)')
    print(f'  Use: X_world = X_pixel * {df["scale_x"].mean():.6f}')
    print(f'  Use: Y_world = Y_pixel * {df["scale_y"].mean():.6f}')
    print('  No homography needed!')
else:
    print('✓ HOMOGRAPHY APPLIED (spatially-varying scale)')
    print('  Scale varies by position - perspective correction applied')
    print('  Need homography matrix or per-position scale lookup')
