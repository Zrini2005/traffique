import pandas as pd

df = pd.read_csv('output/test_homography_200.csv')

# Filter to valid Y range (road region)
valid = df[(df['Y_pixel'] >= 850) & (df['Y_pixel'] <= 1100)]

print('='*70)
print('  FILTERING TO VALID ROAD REGION')
print('='*70)

print(f'\nTotal trajectory points: {len(df)}')
print(f'Points in valid Y range (850-1100px): {len(valid)} ({len(valid)/len(df)*100:.1f}%)')

if len(valid) > 0:
    print(f'\n📊 World Coordinates (filtered to road region):')
    print(f'   X_world: {valid["X_world"].min():.2f} to {valid["X_world"].max():.2f}m')
    print(f'   Y_world: {valid["Y_world"].min():.2f} to {valid["Y_world"].max():.2f}m')
    
    df_friend = pd.read_csv('output/D2F1_lclF.csv')
    print(f'\n📊 Friend\'s World Coordinates (reference):')
    print(f'   X_world: {df_friend["X_world"].min():.2f} to {df_friend["X_world"].max():.2f}m')
    print(f'   Y_world: {df_friend["Y_world"].min():.2f} to {df_friend["Y_world"].max():.2f}m')
    
    # Check match
    x_match = abs(valid["X_world"].mean() - df_friend["X_world"].mean()) < 50
    y_match = abs(valid["Y_world"].mean() - df_friend["Y_world"].mean()) < 5
    
    print(f'\n✅ Coordinate Match: {"YES!" if x_match and y_match else "Needs adjustment"}')
    
    # Save filtered version
    valid.to_csv('output/test_homography_200_filtered.csv', index=False)
    print(f'\n💾 Saved filtered data to: output/test_homography_200_filtered.csv')

print('\n' + '='*70)
