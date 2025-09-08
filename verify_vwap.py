import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('datadoge.csv')

# Get the last 48 candles (excluding header)
last_48 = df.tail(48)

print("Last 48 candles:")
print(f"First candle: {last_48.iloc[0]['DateTime']} - Close: {last_48.iloc[0]['Close']}")
print(f"Last candle: {last_48.iloc[-1]['DateTime']} - Close: {last_48.iloc[-1]['Close']}")

# Calculate VWAP manually
volume_price = last_48['Close'] * last_48['Volume']
total_volume_price = volume_price.sum()
total_volume = last_48['Volume'].sum()
vwap = total_volume_price / total_volume

print(f"\nVWAP Calculation:")
print(f"Total Volume * Price: {total_volume_price:,.2f}")
print(f"Total Volume: {total_volume:,.2f}")
print(f"VWAP: {vwap:.6f}")

print(f"\nCurrent Price: {last_48.iloc[-1]['Close']:.6f}")
print(f"Price vs VWAP: {((last_48.iloc[-1]['Close'] / vwap) - 1) * 100:.3f}%")

# Also check the rolling VWAP calculation
last_48['volume_price'] = last_48['Close'] * last_48['Volume']
last_48['rolling_vwap'] = last_48['volume_price'].rolling(window=48).sum() / last_48['Volume'].rolling(window=48).sum()

print(f"\nRolling VWAP (last value): {last_48['rolling_vwap'].iloc[-1]:.6f}")
print(f"Rolling VWAP (second to last): {last_48['rolling_vwap'].iloc[-2]:.6f}")

# Check if there are any NaN values
print(f"\nNaN values in rolling VWAP: {last_48['rolling_vwap'].isna().sum()}")
print(f"Non-NaN values: {last_48['rolling_vwap'].notna().sum()}")
