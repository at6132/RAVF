import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('datadoge.csv')

# Get the last 100 candles
last_100 = df.tail(100)

print(f"Loaded {len(last_100)} candles")
print(f"First candle: {last_100.iloc[0]['DateTime']} - Volume: {last_100.iloc[0]['Volume']}")
print(f"Last candle: {last_100.iloc[-1]['DateTime']} - Volume: {last_100.iloc[-1]['Volume']}")

# Calculate z-Volume for the last 48 candles (like the live script should)
last_48 = last_100.tail(48)

print(f"\nLast 48 candles volume data:")
print(f"Current volume: {last_48.iloc[-1]['Volume']:.2f}")
print(f"Volume range: {last_48['Volume'].min():.2f} - {last_48['Volume'].max():.2f}")
print(f"Volume mean: {last_48['Volume'].mean():.2f}")

# Calculate z-Volume manually
volume_mean = last_48['Volume'].rolling(window=48).mean()
zvol = last_48['Volume'] / volume_mean

print(f"\nz-Volume calculation:")
print(f"Volume mean (48-period): {volume_mean.iloc[-1]:.2f}")
print(f"Current z-Volume: {zvol.iloc[-1]:.2f}")

# Check for NaN values
print(f"\nNaN values in z-Volume: {zvol.isna().sum()}")
print(f"Non-NaN values: {zvol.notna().sum()}")

# Show some recent z-Volume values
print(f"\nRecent z-Volume values:")
for i in range(-5, 0):
    if not pd.isna(zvol.iloc[i]):
        print(f"  Candle {i}: Volume={last_48.iloc[i]['Volume']:.2f}, z-Vol={zvol.iloc[i]:.2f}")

# Test the rolling calculation step by step
print(f"\nStep-by-step z-Volume calculation:")
print(f"1. Current volume: {last_48.iloc[-1]['Volume']:.2f}")
print(f"2. 48-period volume mean: {last_48['Volume'].mean():.2f}")
print(f"3. z-Volume = {last_48.iloc[-1]['Volume']:.2f} / {last_48['Volume'].mean():.2f} = {last_48.iloc[-1]['Volume'] / last_48['Volume'].mean():.2f}")

# Check if the issue is with the rolling window
print(f"\nRolling window test:")
for window in [10, 20, 30, 40, 48]:
    if len(last_48) >= window:
        vol_mean = last_48['Volume'].rolling(window=window).mean()
        zvol_test = last_48['Volume'] / vol_mean
        print(f"Window {window}: z-Vol = {zvol_test.iloc[-1]:.2f} (NaN count: {zvol_test.isna().sum()})")
