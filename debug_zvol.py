import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('datadoge.csv')

print(f"Total candles in CSV: {len(df)}")
print(f"First candle: {df.iloc[0]['DateTime']}")
print(f"Last candle: {df.iloc[-1]['DateTime']}")

# Test with different amounts of historical data
for num_candles in [50, 100, 200, 300]:
    print(f"\n=== Testing with {num_candles} candles ===")
    
    # Get the last N candles
    test_data = df.tail(num_candles)
    
    # Calculate z-Volume
    volume_mean = test_data['Volume'].rolling(window=48).mean()
    zvol = test_data['Volume'] / volume_mean
    
    print(f"Data length: {len(test_data)}")
    print(f"z-Volume NaN count: {zvol.isna().sum()}")
    print(f"z-Volume valid count: {zvol.notna().sum()}")
    print(f"Last z-Volume value: {zvol.iloc[-1]:.2f}")
    print(f"Last 5 z-Volume values: {zvol.tail().tolist()}")
    
    # Check if the issue is with the rolling window
    if len(test_data) >= 48:
        print(f"48-period volume mean: {test_data['Volume'].tail(48).mean():.2f}")
        print(f"Current volume: {test_data['Volume'].iloc[-1]:.2f}")
        print(f"Manual z-Vol: {test_data['Volume'].iloc[-1] / test_data['Volume'].tail(48).mean():.2f}")

# Test the exact same logic as the live script
print(f"\n=== Testing live script logic ===")

# Simulate loading 100 candles
last_100 = df.tail(100)
print(f"Loaded {len(last_100)} candles")

# Create DataFrame like the live script does
hist_df = pd.DataFrame({
    'close': last_100['Close'].values,
    'volume': last_100['Volume'].values
})

print(f"Hist DF shape: {hist_df.shape}")
print(f"Hist DF columns: {list(hist_df.columns)}")

# Calculate z-Volume like the live script
volume_mean = hist_df['volume'].rolling(window=48).mean()
hist_df['zvol'] = hist_df['volume'] / volume_mean

print(f"z-Volume calculation:")
print(f"Volume mean shape: {volume_mean.shape}")
print(f"z-Volume shape: {hist_df['zvol'].shape}")
print(f"z-Volume NaN count: {hist_df['zvol'].isna().sum()}")
print(f"z-Volume valid count: {hist_df['zvol'].notna().sum()}")
print(f"Last z-Volume: {hist_df['zvol'].iloc[-1]:.2f}")

# Show the last few values
print(f"Last 10 z-Volume values:")
for i in range(-10, 0):
    val = hist_df['zvol'].iloc[i]
    if pd.isna(val):
        print(f"  Index {i}: NaN")
    else:
        print(f"  Index {i}: {val:.2f}")
