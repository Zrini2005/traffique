import pandas as pd

df = pd.read_csv('output/test_homography_200.csv')

print('\n' + '='*70)
print('  WORLD COORDINATE VALIDATION')
print('='*70)

print('\n📊 Your Results (200 frames):')
print(f'   X_world: {df["X_world"].min():.2f} to {df["X_world"].max():.2f} meters')
print(f'   Y_world: {df["Y_world"].min():.2f} to {df["Y_world"].max():.2f} meters')
print(f'   Range: X={df["X_world"].max() - df["X_world"].min():.2f}m, Y={df["Y_world"].max() - df["Y_world"].min():.2f}m')

df_friend = pd.read_csv('output/D2F1_lclF.csv')
print('\n📊 Friend\'s Results (D2F1_lclF.csv):')
print(f'   X_world: {df_friend["X_world"].min():.2f} to {df_friend["X_world"].max():.2f} meters')
print(f'   Y_world: {df_friend["Y_world"].min():.2f} to {df_friend["Y_world"].max():.2f} meters')
print(f'   Range: X={df_friend["X_world"].max() - df_friend["X_world"].min():.2f}m, Y={df_friend["Y_world"].max() - df_friend["Y_world"].min():.2f}m')

# Check if in range
x_match = df_friend["X_world"].min() - 10 < df["X_world"].min() < df_friend["X_world"].max() + 10
y_match = df_friend["Y_world"].min() - 2 < df["Y_world"].min() < df_friend["Y_world"].max() + 2

print(f'\n✅ Match: {"YES - Coordinates in expected range!" if x_match and y_match else "NO - Outside expected range"}')

print('\n' + '='*70)
