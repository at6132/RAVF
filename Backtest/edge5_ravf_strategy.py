import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class Edge5RAVFStrategy:
    def __init__(self, lookback_days=180):
        self.lookback_days = lookback_days
        self.signals = []
        
    def calculate_returns(self, df):
        """Calculate log returns"""
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        return df
    
    def calculate_realized_volatility(self, df, window=48):
        """Calculate rolling realized volatility (std of returns)"""
        df['rv'] = df['returns'].rolling(window=window).std()
        return df
    
    def calculate_skew_kurt(self, df, window=48):
        """Calculate rolling skewness and kurtosis"""
        df['skew'] = df['returns'].rolling(window=window).skew()
        df['kurt'] = df['returns'].rolling(window=window).kurt()
        return df
    
    def calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=window).mean()
        return df
    
    def calculate_clv(self, df):
        """Calculate Close Location Value"""
        df['clv'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['clv'] = df['clv'].clip(-1, 1)
        return df
    
    def calculate_avwap(self, df):
        """Calculate Daily Anchored VWAP"""
        df['date'] = pd.to_datetime(df['timestamp'], unit='ns').dt.date
        df['vwap'] = (df['close'] * df['volume']).groupby(df['date']).cumsum() / df['volume'].groupby(df['date']).cumsum()
        return df
    
    def calculate_minute_of_day(self, df):
        """Calculate minute of day slot (0-287 for 5-minute bars)"""
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ns')
        df['minute_of_day'] = df['datetime'].dt.hour * 12 + df['datetime'].dt.minute // 5
        return df
    
    def calculate_relative_volume(self, df, lookback_days=30):
        """Calculate relative volume using minute-of-day baseline"""
        df['rvol'] = 1.0  # Default value
        
        # For each minute of day, calculate median volume over last 30 days
        for minute in range(288):  # 0-287 for 5-minute bars
            mask = df['minute_of_day'] == minute
            if mask.sum() > 0:
                # Get last 30 days of data for this minute
                minute_data = df[mask].copy()
                minute_data['date'] = pd.to_datetime(minute_data['timestamp'], unit='ns').dt.date
                
                # Calculate median volume for each date
                daily_volumes = minute_data.groupby('date')['volume'].sum()
                
                if len(daily_volumes) >= 7:  # Need at least a week of data
                    # Use last 30 days or available data
                    recent_volumes = daily_volumes.tail(min(30, len(daily_volumes)))
                    median_vol = recent_volumes.median()
                    
                    if median_vol > 0:
                        df.loc[mask, 'rvol'] = df.loc[mask, 'volume'] / median_vol
        
        # Calculate z-score of log relative volume using MAD
        df['zvol'] = 0.0
        log_rvol = np.log(df['rvol'].replace(0, 1))
        mad = np.median(np.abs(log_rvol - np.median(log_rvol)))
        if mad > 0:
            df['zvol'] = (log_rvol - np.median(log_rvol)) / (1.4826 * mad)
        
        return df
    
    def detect_compression(self, df):
        """Detect 3-bar compression patterns"""
        df['compression'] = False
        df['comp_high'] = np.nan
        df['comp_low'] = np.nan
        
        for i in range(2, len(df)):
            if i < 2:
                continue
                
            # Check if last 3 bars are inside/narrow
            bar1, bar2, bar3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
            
            # Check if ranges are narrow (< 0.6 * ATR)
            narrow1 = (bar1['high'] - bar1['low']) < 0.6 * bar1['atr']
            narrow2 = (bar2['high'] - bar2['low']) < 0.6 * bar2['atr']
            narrow3 = (bar3['high'] - bar3['low']) < 0.6 * bar3['atr']
            
            # Check if bars are inside (highs decreasing, lows increasing)
            inside1 = (bar2['high'] <= bar1['high']) and (bar2['low'] >= bar1['low'])
            inside2 = (bar3['high'] <= bar2['high']) and (bar3['low'] >= bar2['low'])
            
            if narrow1 and narrow2 and narrow3 and inside1 and inside2:
                df.loc[df.index[i], 'compression'] = True
                df.loc[df.index[i], 'comp_high'] = max(bar1['high'], bar2['high'], bar3['high'])
                df.loc[df.index[i], 'comp_low'] = min(bar1['low'], bar2['low'], bar3['low'])
        
        return df
    
    def detect_regime(self, df):
        """Detect market regime based on RV, skew, kurt, and zVol - More aggressive"""
        df['regime'] = 'Neutral'
        
        # Calculate percentiles for RV over rolling 60-day lookback
        df['rv_percentile'] = df['rv'].rolling(window=288).rank(pct=True)  # 288 = 24 hours
        
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['rv']) or pd.isna(df.iloc[i]['skew']) or pd.isna(df.iloc[i]['kurt']):
                continue
                
            rv = df.iloc[i]['rv_percentile']
            skew = df.iloc[i]['skew']
            kurt = df.iloc[i]['kurt']
            zvol = df.iloc[i]['zvol']
            
            # Trend regime - More relaxed
            if (rv >= 0.6 and (skew > 0.3 or kurt > 3.0) and zvol > 1.0):
                df.loc[df.index[i], 'regime'] = 'Trend'
            # Mean-reversion regime - More relaxed
            elif (rv <= 0.5 and kurt < 3.0):
                df.loc[df.index[i], 'regime'] = 'Mean-reversion'
            # Neutral regime - Add some signals here too
            elif (rv >= 0.4 and rv <= 0.6):
                df.loc[df.index[i], 'regime'] = 'Neutral'
        
        return df
    
    def generate_signals(self, df):
        """Generate entry signals based on RAVF strategy"""
        df['signal'] = 0  # 0: no signal, 1: long, -1: short
        df['signal_type'] = ''
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        
        for i in range(1, len(df)):
            if i < 14:  # Need enough data for ATR
                continue
                
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # A) Trend-expansion "fracture" long - More relaxed
            if (current['regime'] == 'Trend' and 
                prev['compression'] and
                current['close'] > prev['comp_high'] and
                current['clv'] >= 0.3 and
                current['zvol'] >= 0.8 and
                current['vwap'] > prev['vwap']):
                
                # Noise filter: avoid hyper-chop
                if i >= 11:
                    recent_ranges = df.iloc[i-11:i]['high'] - df.iloc[i-11:i]['low']
                    if recent_ranges.median() / current['atr'] <= 0.85:
                        df.loc[df.index[i], 'signal'] = 1
                        df.loc[df.index[i], 'signal_type'] = 'Trend_Long'
                        df.loc[df.index[i], 'entry_price'] = current['close']
                        df.loc[df.index[i], 'stop_loss'] = prev['comp_low']
            
            # A) Trend-expansion "fracture" short - More relaxed
            elif (current['regime'] == 'Trend' and 
                  prev['compression'] and
                  current['close'] < prev['comp_low'] and
                  current['clv'] <= -0.3 and
                  current['zvol'] >= 0.8 and
                  current['vwap'] < prev['vwap']):
                
                if i >= 11:
                    recent_ranges = df.iloc[i-11:i]['high'] - df.iloc[i-11:i]['low']
                    if recent_ranges.median() / current['atr'] <= 0.85:
                        df.loc[df.index[i], 'signal'] = -1
                        df.loc[df.index[i], 'signal_type'] = 'Trend_Short'
                        df.loc[df.index[i], 'entry_price'] = current['close']
                        df.loc[df.index[i], 'stop_loss'] = prev['comp_high']
            
            # B) Mean-reversion AVWAP snapback long - More relaxed
            elif (current['regime'] == 'Mean-reversion' and
                  current['close'] <= current['vwap'] - 0.6 * current['atr'] and
                  current['clv'] <= -0.4 and
                  current['zvol'] >= 1.2):
                
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_type'] = 'MeanRev_Long'
                df.loc[df.index[i], 'entry_price'] = current['close']
                df.loc[df.index[i], 'stop_loss'] = current['vwap'] - 1.0 * current['atr']
            
            # B) Mean-reversion AVWAP snapback short - More relaxed
            elif (current['regime'] == 'Mean-reversion' and
                  current['close'] >= current['vwap'] + 0.6 * current['atr'] and
                  current['clv'] >= 0.4 and
                  current['zvol'] >= 1.2):
                
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'signal_type'] = 'MeanRev_Short'
                df.loc[df.index[i], 'entry_price'] = current['close']
                df.loc[df.index[i], 'stop_loss'] = current['vwap'] + 1.0 * current['atr']
            
            # C) Neutral regime signals - New signal type
            elif (current['regime'] == 'Neutral' and
                  current['zvol'] >= 1.0):
                
                # Simple momentum signals
                if (current['close'] > current['vwap'] and 
                    current['clv'] >= 0.3 and
                    current['close'] > prev['close']):
                    
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'signal_type'] = 'Neutral_Long'
                    df.loc[df.index[i], 'entry_price'] = current['close']
                    df.loc[df.index[i], 'stop_loss'] = current['vwap'] - 0.8 * current['atr']
                
                elif (current['close'] < current['vwap'] and 
                      current['clv'] <= -0.3 and
                      current['close'] < prev['close']):
                    
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_type'] = 'Neutral_Short'
                    df.loc[df.index[i], 'entry_price'] = current['close']
                    df.loc[df.index[i], 'stop_loss'] = current['vwap'] + 0.8 * current['atr']
            
            # D) Volume breakout signals - Additional signal type
            elif (current['zvol'] >= 1.5 and
                  abs(current['clv']) >= 0.5):
                
                if current['close'] > prev['close']:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'signal_type'] = 'Volume_Long'
                    df.loc[df.index[i], 'entry_price'] = current['close']
                    df.loc[df.index[i], 'stop_loss'] = current['close'] - 0.5 * current['atr']
                
                elif current['close'] < prev['close']:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_type'] = 'Volume_Short'
                    df.loc[df.index[i], 'entry_price'] = current['close']
                    df.loc[df.index[i], 'stop_loss'] = current['close'] + 0.5 * current['atr']
            
            # E) Simple momentum signals - High frequency
            elif (abs(current['clv']) >= 0.7 and
                  current['zvol'] >= 0.5):
                
                if current['close'] > prev['close'] and current['close'] > current['vwap']:
                    df.loc[df.index[i], 'signal'] = 1
                    df.loc[df.index[i], 'signal_type'] = 'Momentum_Long'
                    df.loc[df.index[i], 'entry_price'] = current['close']
                    df.loc[df.index[i], 'stop_loss'] = current['close'] - 0.3 * current['atr']
                
                elif current['close'] < prev['close'] and current['close'] < current['vwap']:
                    df.loc[df.index[i], 'signal'] = -1
                    df.loc[df.index[i], 'signal_type'] = 'Momentum_Short'
                    df.loc[df.index[i], 'entry_price'] = current['close']
                    df.loc[df.index[i], 'stop_loss'] = current['close'] + 0.3 * current['atr']
        
        return df
    
    def analyze_excursions(self, df, bars_after=5):
        """Analyze price excursions after each signal"""
        excursions = []
        
        for i in range(len(df) - bars_after):
            if df.iloc[i]['signal'] != 0:
                signal = df.iloc[i]
                entry_price = signal['entry_price']
                signal_type = signal['signal_type']
                
                # Get next 5 bars
                future_bars = df.iloc[i+1:i+1+bars_after]
                
                if len(future_bars) == bars_after:
                    # Calculate max adverse and max favorable moves
                    if signal['signal'] == 1:  # Long position
                        prices = future_bars['high']
                        max_favorable = (prices.max() - entry_price) / entry_price * 100
                        
                        # For max adverse, only count trades that went down then back up into profit
                        prices_with_entry = pd.concat([pd.Series([entry_price]), future_bars['low']])
                        cumulative_return = (prices_with_entry - entry_price) / entry_price
                        
                        # Find if it went negative then back to positive
                        went_negative = (cumulative_return < 0).any()
                        came_back_positive = (cumulative_return > 0).any()
                        
                        if went_negative and came_back_positive:
                            max_adverse = (prices_with_entry.min() - entry_price) / entry_price * 100
                        else:
                            max_adverse = (prices_with_entry.min() - entry_price) / entry_price * 100
                    
                    else:  # Short position
                        prices = future_bars['low']
                        max_favorable = (entry_price - prices.min()) / entry_price * 100
                        
                        prices_with_entry = pd.concat([pd.Series([entry_price]), future_bars['high']])
                        cumulative_return = (entry_price - prices_with_entry) / entry_price
                        
                        went_negative = (cumulative_return < 0).any()
                        came_back_positive = (cumulative_return > 0).any()
                        
                        if went_negative and came_back_positive:
                            max_adverse = (entry_price - prices_with_entry.max()) / entry_price * 100
                        else:
                            max_adverse = (entry_price - prices_with_entry.max()) / entry_price * 100
                    
                    excursions.append({
                        'timestamp': signal['datetime'],
                        'signal_type': signal_type,
                        'entry_price': entry_price,
                        'regime': signal['regime'],
                        'max_adverse': max_adverse,
                        'max_favorable': max_favorable,
                        'went_negative_then_profit': went_negative and came_back_positive
                    })
        
        return pd.DataFrame(excursions)
    
    def run_strategy(self, df):
        """Run the complete RAVF strategy"""
        print("Running EDGE-5 RAVF Strategy...")
        
        # Apply all calculations
        df = self.calculate_returns(df)
        df = self.calculate_realized_volatility(df)
        df = self.calculate_skew_kurt(df)
        df = self.calculate_atr(df)
        df = self.calculate_clv(df)
        df = self.calculate_avwap(df)
        df = self.calculate_minute_of_day(df)
        df = self.calculate_relative_volume(df)
        df = self.detect_compression(df)
        df = self.detect_regime(df)
        df = self.generate_signals(df)
        
        # Analyze excursions
        excursions = self.analyze_excursions(df)
        
        return df, excursions

def main():
    # Directory containing 5-minute candle data
    data_dir = "5m"
    
    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} not found!")
        return
    
    # Get all CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    csv_files.sort()
    
    # Initialize strategy
    strategy = Edge5RAVFStrategy()
    
    all_excursions = []
    
    print("=" * 80)
    print("EDGE-5 RAVF STRATEGY EXCURSION ANALYSIS")
    print("=" * 80)
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file}...")
        file_path = os.path.join(data_dir, csv_file)
        
        # Read data
        df = pd.read_csv(file_path, header=None, names=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'
        ])
        
        # Run strategy
        df_with_signals, excursions = strategy.run_strategy(df)
        
        # Count signals
        signal_count = len(excursions)
        print(f"Generated {signal_count} signals")
        
        if signal_count > 0:
            all_excursions.append(excursions)
    
    if all_excursions:
        # Combine all excursions
        combined_excursions = pd.concat(all_excursions, ignore_index=True)
        
        print(f"\n" + "=" * 80)
        print("EXCURSION ANALYSIS RESULTS")
        print("=" * 80)
        print(f"Total signals: {len(combined_excursions)}")
        
        # Overall statistics
        print(f"\nOverall Statistics (5 bars after entry):")
        print(f"Average Max Adverse Move: {combined_excursions['max_adverse'].mean():.2f}%")
        print(f"Average Max Favorable Move: {combined_excursions['max_favorable'].mean():.2f}%")
        
        # Filter for trades that went negative then back to profit
        profitable_recovery = combined_excursions[combined_excursions['went_negative_then_profit']]
        if len(profitable_recovery) > 0:
            print(f"\nTrades that went negative then back to profit: {len(profitable_recovery)}")
            print(f"Average Max Adverse Move (recovery trades): {profitable_recovery['max_adverse'].mean():.2f}%")
            print(f"Average Max Favorable Move (recovery trades): {profitable_recovery['max_favorable'].mean():.2f}%")
        
        # By signal type
        print(f"\nBy Signal Type:")
        for signal_type in combined_excursions['signal_type'].unique():
            type_data = combined_excursions[combined_excursions['signal_type'] == signal_type]
            print(f"{signal_type}: {len(type_data)} signals")
            print(f"  Avg Max Adverse: {type_data['max_adverse'].mean():.2f}%")
            print(f"  Avg Max Favorable: {type_data['max_favorable'].mean():.2f}%")
        
        # By regime
        print(f"\nBy Market Regime:")
        for regime in combined_excursions['regime'].unique():
            regime_data = combined_excursions[combined_excursions['regime'] == regime]
            print(f"{regime}: {len(regime_data)} signals")
            print(f"  Avg Max Adverse: {regime_data['max_adverse'].mean():.2f}%")
            print(f"  Avg Max Favorable: {regime_data['max_favorable'].mean():.2f}%")
        
        # Save results
        output_file = "edge5_ravf_excursions.csv"
        combined_excursions.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Show top 10 most favorable moves
        print(f"\nTop 10 Most Favorable Moves:")
        top_moves = combined_excursions.nlargest(10, 'max_favorable')
        for i, move in top_moves.iterrows():
            print(f"{i+1:2d}. {move['timestamp']} - {move['signal_type']} - {move['max_favorable']:.2f}% favorable, {move['max_adverse']:.2f}% adverse")

if __name__ == "__main__":
    main()
