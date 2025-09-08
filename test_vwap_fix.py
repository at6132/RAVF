import pandas as pd
import numpy as np

# Test the VWAP calculation with the last 100 candles
df = pd.read_csv('datadoge.csv')

# Get the last 100 candles
last_100 = df.tail(100)

print(f"Loaded {len(last_100)} candles")
print(f"First candle: {last_100.iloc[0]['DateTime']} - Close: {last_100.iloc[0]['Close']}")
print(f"Last candle: {last_100.iloc[-1]['DateTime']} - Close: {last_100.iloc[-1]['Close']}")

# Calculate VWAP for the last 48 candles (like the live script should)
last_48 = last_100.tail(48)
volume_price = last_48['Close'] * last_48['Volume']
total_volume_price = volume_price.sum()
total_volume = last_48['Volume'].sum()
vwap = total_volume_price / total_volume

print(f"\nVWAP Calculation (last 48 candles):")
print(f"VWAP: {vwap:.6f}")
print(f"Current Price: {last_48.iloc[-1]['Close']:.6f}")
print(f"Price vs VWAP: {((last_48.iloc[-1]['Close'] / vwap) - 1) * 100:.3f}%")

# Test rolling VWAP calculation
last_100['volume_price'] = last_100['Close'] * last_100['Volume']
last_100['rolling_vwap'] = last_100['volume_price'].rolling(window=48).sum() / last_100['Volume'].rolling(window=48).sum()

print(f"\nRolling VWAP (last value): {last_100['rolling_vwap'].iloc[-1]:.6f}")
print(f"Rolling VWAP (second to last): {last_100['rolling_vwap'].iloc[-2]:.6f}")

# Check if there are any NaN values
print(f"\nNaN values in rolling VWAP: {last_100['rolling_vwap'].isna().sum()}")
print(f"Non-NaN values: {last_100['rolling_vwap'].notna().sum()}")
