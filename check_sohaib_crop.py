import pandas as pd

df = pd.read_csv('output/D2F1_lclF.csv')

print("="*70)
print("  ANALYZING SOHAIB'S CSV - DID HE CROP?")
print("="*70)

x_min = df['X_pixel'].min()
x_max = df['X_pixel'].max()
y_min = df['Y_pixel'].min()
y_max = df['Y_pixel'].max()

x_range = x_max - x_min
y_range = y_max - y_min

print(f"\n📐 Pixel Coordinate Ranges:")
print(f"   X_pixel: {x_min:.1f} to {x_max:.1f} (width: {x_range:.1f}px)")
print(f"   Y_pixel: {y_min:.1f} to {y_max:.1f} (height: {y_range:.1f}px)")

print(f"\n🎥 Video Dimensions:")
print(f"   Full frame: 3840 x 2160 pixels")

print(f"\n🔍 Analysis:")
print(f"   Sohaib's X coverage: {x_range:.0f}px / 3840px = {x_range/3840*100:.1f}%")
print(f"   Sohaib's Y coverage: {y_range:.0f}px / 2160px = {y_range/2160*100:.1f}%")

print(f"\n✅ Conclusion:")
if y_range < 300:
    print(f"   YES - Sohaib CROPPED to road region!")
    print(f"   He only tracked a {y_range:.0f}px tall strip (Y: {y_min:.0f}-{y_max:.0f})")
    print(f"   This is {y_min:.0f}px from the top of the frame")
else:
    print(f"   NO - Sohaib used full frame or large region")

print("\n" + "="*70)
