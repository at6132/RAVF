import pandas as pd
import numpy as np
import os
from datetime import datetime

class IndicatorTester:
    """Test script to verify indicator calculations match backtester"""
    
    def __init__(self):
        self.candles = []
        
    def _calculate_atr_safe(self, df, window=14):
        """Calculate ATR with fallback for insufficient data"""
        if len(df) < 2:
            return abs(df['high'] - df['low'])
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=min(window, len(df))).mean()
        return df['atr']
    
    def _calculate_clv_safe(self, df):
        """Calculate CLV with fallback for insufficient data"""
        if len(df) == 0:
            return pd.Series([0.0])
        
        # CLV = ((close - low) - (high - close)) / (high - low)
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['clv'] = clv.clip(-1, 1)
        return df['clv']
    
    def _calculate_vwap_safe(self, df, window=48):
        """Calculate VWAP with fallback for insufficient data"""
        if len(df) == 0:
            return pd.Series([0.0])
        
        volume_price = df['close'] * df['volume']
        rolling_vwap = volume_price.rolling(window=min(window, len(df))).sum() / df['volume'].rolling(window=min(window, len(df))).sum()
        df['vwap'] = rolling_vwap
        return df['vwap']
    
    def _calculate_relative_volume_safe(self, df, lookback=48):
        """Calculate relative volume with fallback for insufficient data"""
        if len(df) == 0:
            return pd.Series([1.0])
        
        if len(df) >= lookback:
            df['zvol'] = df['volume'] / df['volume'].rolling(window=lookback).mean()
        else:
            df['zvol'] = 1.0
        return df['zvol']
    
    def calculate_entropy(self, df, window=20):
        """Calculate entropy of returns (matches backtester exactly)"""
        def entropy_calc(returns):
            if len(returns) < 2:
                return 0.0
            returns = returns.dropna()
            if len(returns) < 2:
                return 0.0
            try:
                # Calculate entropy as -sum(p * log(p)) where p is normalized returns
                abs_returns = np.abs(returns)
                if abs_returns.sum() == 0:
                    return 0.0
                p = abs_returns / abs_returns.sum()
                p = p[p > 0]  # Remove zeros
                return -np.sum(p * np.log(p + 1e-10))
            except:
                return 0.0
        
        df['entropy'] = df['returns'].rolling(window=window).apply(entropy_calc, raw=False)
        return df['entropy']
    
    def process_candle(self, candle_data):
        """Process a single candle and calculate indicators"""
        # Convert candle data to DataFrame format
        df = pd.DataFrame([candle_data], columns=[
            'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'
        ])
        
        # Convert numeric data
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Add to historical data
        self.candles.append(candle_data)
        
        # Keep only last 100 candles
        if len(self.candles) > 100:
            self.candles = self.candles[-100:]
        
        # Calculate indicators
        if len(self.candles) > 0:
            # Create DataFrame from all historical candles for rolling indicators
            hist_df = pd.DataFrame(self.candles, columns=[
                'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'
            ])
            
            # Convert numeric data
            hist_df['open'] = pd.to_numeric(hist_df['open'], errors='coerce')
            hist_df['high'] = pd.to_numeric(hist_df['high'], errors='coerce')
            hist_df['low'] = pd.to_numeric(hist_df['low'], errors='coerce')
            hist_df['close'] = pd.to_numeric(hist_df['close'], errors='coerce')
            hist_df['volume'] = pd.to_numeric(hist_df['volume'], errors='coerce')
            
            # Calculate rolling indicators on historical data
            hist_df['atr'] = self._calculate_atr_safe(hist_df)
            hist_df['vwap'] = self._calculate_vwap_safe(hist_df)
            hist_df['zvol'] = self._calculate_relative_volume_safe(hist_df)
            
            # Get the latest values for current candle
            atr = hist_df['atr'].iloc[-1] if len(hist_df) > 0 else 0.0
            vwap = hist_df['vwap'].iloc[-1] if len(hist_df) > 0 else df['close'].iloc[0]
            zvol = hist_df['zvol'].iloc[-1] if len(hist_df) > 0 else 1.0
        else:
            atr = 0.0
            vwap = df['close'].iloc[0]
            zvol = 1.0
        
        # Calculate CLV on current candle only
        df['clv'] = self._calculate_clv_safe(df)
        clv = df['clv'].iloc[0]
        
        # Calculate entropy using historical data (matches backtester)
        if len(self.candles) > 0:
            hist_df = pd.DataFrame(self.candles, columns=[
                'timestamp', 'datetime', 'open', 'high', 'low', 'close', 'volume'
            ])
            hist_df['close'] = pd.to_numeric(hist_df['close'], errors='coerce')
            hist_df['returns'] = np.log(hist_df['close'] / hist_df['close'].shift(1))
            hist_df['entropy'] = self.calculate_entropy(hist_df)
            entropy = hist_df['entropy'].iloc[-1] if len(hist_df) > 0 else 0.0
        else:
            entropy = 0.0
        
        # Calculate VWAP bands
        upper_band = vwap + (0.6 * atr)
        lower_band = vwap - (0.6 * atr)
        
        # Calculate price vs VWAP percentage
        price_vs_vwap = ((df['close'].iloc[0] - vwap) / vwap) * 100
        
        # Check signal conditions
        close_price = df['close'].iloc[0]
        
        # LONG signal conditions
        long_vwap_band = close_price <= lower_band
        long_clv = clv <= -0.4
        long_volume = zvol >= 1.2
        long_signal = long_vwap_band and long_clv and long_volume
        
        # SHORT signal conditions
        short_vwap_band = close_price >= upper_band
        short_clv = clv >= 0.4
        short_volume = zvol >= 1.2
        short_signal = short_vwap_band and short_clv and short_volume
        
        return {
            'timestamp': candle_data[1],
            'price': close_price,
            'vwap': vwap,
            'atr': atr,
            'clv': clv,
            'zvol': zvol,
            'entropy': entropy,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'price_vs_vwap_pct': price_vs_vwap,
            'long_signal': long_signal,
            'short_signal': short_signal,
            'long_conditions': {
                'vwap_band': long_vwap_band,
                'clv': long_clv,
                'volume': long_volume
            },
            'short_conditions': {
                'vwap_band': short_vwap_band,
                'clv': short_clv,
                'volume': short_volume
            }
        }
    
    def test_with_csv(self, csv_file='datadoge.csv'):
        """Test indicators using CSV data"""
        if not os.path.exists(csv_file):
            print(f"❌ CSV file not found: {csv_file}")
            return
        
        print(f"🔍 Testing indicators with {csv_file}")
        print("="*80)
        
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        # Skip header
        data_lines = lines[1:]
        
        for i, line in enumerate(data_lines):
            candle_data = line.strip().split(',')
            if len(candle_data) >= 7:
                result = self.process_candle(candle_data)
                
                # Print every 10th candle or if there's a signal
                if i % 10 == 0 or result['long_signal'] or result['short_signal']:
                    print(f"\n🕒 Candle {i+1}: {result['timestamp']}")
                    print(f"💰 Price: ${result['price']:.6f} | VWAP: ${result['vwap']:.6f} | Volume: {candle_data[6]}")
                    print(f"📊 ATR: {result['atr']:.6f} | CLV: {result['clv']:.4f} | zVol: {result['zvol']:.2f} | Entropy: {result['entropy']:.4f}")
                    print(f"🎯 VWAP Bands: Upper ${result['upper_band']:.6f} | Lower ${result['lower_band']:.6f}")
                    print(f"📈 Price vs VWAP: {result['price_vs_vwap_pct']:+.3f}%")
                    
                    if result['long_signal']:
                        print("🚨 LONG SIGNAL TRIGGERED!")
                    elif result['short_signal']:
                        print("🚨 SHORT SIGNAL TRIGGERED!")
                    else:
                        print("⏳ No signal")
                        print(f"   LONG: VWAP={result['long_conditions']['vwap_band']}, CLV={result['long_conditions']['clv']}, Vol={result['long_conditions']['volume']}")
                        print(f"   SHORT: VWAP={result['short_conditions']['vwap_band']}, CLV={result['short_conditions']['clv']}, Vol={result['short_conditions']['volume']}")
                    
                    print("-" * 80)

if __name__ == "__main__":
    tester = IndicatorTester()
    tester.test_with_csv()
