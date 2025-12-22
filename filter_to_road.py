import pandas as pd

print("Loading your full trajectory CSV...")
df = pd.read_csv('output/my_d2f1_full.csv')

print(f"Total points: {len(df)}")
print(f"Total vehicles: {df['VehicleID'].nunique()}")

# Filter to valid road region (Y: 850-1100 pixels)
print("\nFiltering to road region (Y_pixel: 850-1100)...")
df_filtered = df[(df['Y_pixel'] >= 850) & (df['Y_pixel'] <= 1100)]

print(f"Filtered points: {len(df_filtered)} ({len(df_filtered)/len(df)*100:.1f}%)")
print(f"Filtered vehicles: {df_filtered['VehicleID'].nunique()}")

# Also filter by minimum trajectory length
vehicle_counts = df_filtered.groupby('VehicleID').size()
valid_vehicles = vehicle_counts[vehicle_counts >= 15].index
df_final = df_filtered[df_filtered['VehicleID'].isin(valid_vehicles)]

print(f"\nAfter min length filter (>=15 frames):")
print(f"Final points: {len(df_final)}")
print(f"Final vehicles: {df_final['VehicleID'].nunique()}")

# Save filtered version
output_file = 'output/my_d2f1_full_filtered.csv'
df_final.to_csv(output_file, index=False)

print(f"\n✅ Saved filtered CSV: {output_file}")

# Quick stats
print(f"\nWorld coordinate ranges:")
print(f"  X_world: {df_final['X_world'].min():.2f} to {df_final['X_world'].max():.2f}m")
print(f"  Y_world: {df_final['Y_world'].min():.2f} to {df_final['Y_world'].max():.2f}m")
