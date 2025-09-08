import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

class Edge5RAVFBacktester:
    """
    EDGE-5 RAVF STRATEGY BACKTESTER
    
    ENTRY STRATEGY:
    ===============
    MEAN-REVERSION AVWAP SNAPBACK:
    - Long: Price <= VWAP - 0.6*ATR + CLV <= -0.4 + zVol >= 1.2
    - Short: Price >= VWAP + 0.6*ATR + CLV >= 0.4 + zVol >= 1.2
    
    EXIT STRATEGY (5-Bar Exit Engine):
    ===================================
    
    1. SCALE-OUT APPROACH (60% + 40%):
       - TP1: Exit 60% of position at first target
       - Runner: Keep 40% for extended gains
    
    2. 5-BAR EXIT ENGINE:
       - Bar 1: Entry
       - Bar 2: Velocity check + Early adverse management
       - Bar 3: Stall filter (scratch if < +0.25% + volume stall)
       - Bar 4: Snap-lock for strong extension + velocity trail adjustment
       - Bar 5: Hard time stop (mandatory exit)
    
    3. EXIT CONDITIONS (in priority order):
       a) 5-Bar Time Stop (mandatory)
       b) Stop Loss (regime-specific levels)
       c) TP1 Hit (60% scale-out)
       d) TP2 Hit (40% runner)
       e) Trailing Stop (after TP1)
       f) Catastrophic Stop (≤ -0.90%)
       g) Stall Filter (Bar 3 volume stall)
    
    4. REGIME-SPECIFIC LEVELS:
       - Mean-reversion: TP1 +0.50%, TP2 +0.95%, SL -0.80%
    
    5. FEES: 0.03% per trade (0.06% round trip)
    """
    
    def __init__(self, initial_capital=10000, fee_per_trade=0.0003):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            fee_per_trade: Fee per trade (0.03% = 0.0003)
        """
        self.initial_capital = initial_capital
        self.fee_per_trade = fee_per_trade
        self.round_trip_fee = fee_per_trade * 2  # 0.06% round trip
        
        # Strategy parameters
        self.VWAP_ATR_MULTIPLIER = 0.6
        self.CLV_LONG_THRESHOLD = -0.4
        self.CLV_SHORT_THRESHOLD = 0.4
        self.VOLUME_THRESHOLD = 1.2
        
        # Mean reversion parameters
        self.MEAN_REV_TP1_LONG = 0.0050  # +0.50%
        self.MEAN_REV_TP2_LONG = 0.0095  # +0.95%
        self.MEAN_REV_SL_LONG = 0.0080   # -0.80%
        
        self.MEAN_REV_TP1_SHORT = 0.0055  # -0.55%
        self.MEAN_REV_TP2_SHORT = 0.0100  # -1.00%
        self.MEAN_REV_SL_SHORT = 0.0080   # +0.80%
        
        # 5-Bar Exit Engine parameters
        self.CATASTROPHIC_STOP_THRESHOLD = 0.0090  # -0.90%
        self.STALL_FILTER_THRESHOLD = 0.0025  # +0.25%
        self.VELOCITY_THRESHOLD = 0.00225  # 0.225%/bar
        self.SNAP_LOCK_THRESHOLD = 0.0080  # +0.80%
        
        # Results storage
        self.trades = []
        self.equity_curve = [initial_capital]
        self.current_equity = initial_capital
        self.position = None
        
        # Daily guardrails tracking
        self.daily_pnl = 0
        self.daily_time_stops = 0
        self.last_signal_time = None
        self.current_date = None
        
        print(f"🚀 Edge-5 RAVF Backtester Initialized")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        print(f"💸 Round-trip fees: {self.round_trip_fee*100:.2f}%")
        print(f"🎯 Strategy: Mean Reversion AVWAP Snapback")
        print(f"📊 Exit Engine: 5-Bar Time Stop System")
    
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
    
    def calculate_vwap(self, df, window=48):
        """Calculate rolling VWAP to avoid look-ahead bias"""
        volume_price = df['close'] * df['volume']
        rolling_vwap = volume_price.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        df['vwap'] = rolling_vwap
        return df
    
    def calculate_relative_volume(self, df, lookback=48):
        """Calculate relative volume (z-score)"""
        df['zvol'] = df['volume'] / df['volume'].rolling(window=lookback).mean()
        return df
    
    def calculate_entropy(self, df, window=20):
        """Calculate entropy of returns"""
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
        return df
    
    def calculate_regime(self, df):
        """Determine market regime (simplified to mean-reversion only)"""
        # For this backtest, we'll force mean-reversion regime
        df['regime'] = 'Mean-reversion'
        return df
    
    def calculate_indicators(self, df):
        """Calculate all indicators"""
        print("📊 Calculating indicators...")
        
        # Basic calculations
        df = self.calculate_returns(df)
        df = self.calculate_realized_volatility(df)
        df = self.calculate_skew_kurt(df)
        df = self.calculate_atr(df)
        df = self.calculate_clv(df)
        df = self.calculate_vwap(df)
        df = self.calculate_relative_volume(df)
        df = self.calculate_entropy(df)
        df = self.calculate_regime(df)
        
        print("✅ Indicators calculated")
        return df
    
    def check_entry_signal(self, row):
        """Check for mean reversion entry signals"""
        if pd.isna(row['vwap']) or pd.isna(row['atr']) or pd.isna(row['clv']) or pd.isna(row['zvol']):
            return None
        
        close_price = row['close']
        vwap = row['vwap']
        atr = row['atr']
        clv = row['clv']
        zvol = row['zvol']
        
        # MEAN-REVERSION AVWAP SNAPBACK LONG
        if (row['regime'] == 'Mean-reversion' and
            close_price <= vwap - self.VWAP_ATR_MULTIPLIER * atr and
            clv <= self.CLV_LONG_THRESHOLD and
            zvol >= self.VOLUME_THRESHOLD):
            
            return 'MeanRev_Long', 1
        
        # MEAN-REVERSION AVWAP SNAPBACK SHORT
        elif (row['regime'] == 'Mean-reversion' and
              close_price >= vwap + self.VWAP_ATR_MULTIPLIER * atr and
              clv >= self.CLV_SHORT_THRESHOLD and
              zvol >= self.VOLUME_THRESHOLD):
            
            return 'MeanRev_Short', -1
        
        return None
    
    def should_skip_signal(self, current_timestamp):
        """Check daily guardrails"""
        current_date = current_timestamp.date()
        
        # Reset daily tracking if new day
        if self.current_date != current_date:
            self.daily_pnl = 0
            self.daily_time_stops = 0
        
        self.current_date = current_date
        
        # Check if we should stop taking signals
        if self.daily_pnl <= -1.5 * 0.0085 or self.daily_time_stops >= 3:  # -1.5R or 3 time-stops
            # Check if 6 hours have passed since last signal
            if self.last_signal_time:
                time_diff = (current_timestamp - self.last_signal_time).total_seconds()
                if time_diff < 6 * 3600:
                    return True
        
        return False
    
    def create_position(self, row, signal_type, direction):
        """Create new position with mean reversion parameters"""
        position = {
            'entry_time': row['timestamp'],
            'entry_price': row['close'],
            'signal_type': signal_type,
            'direction': direction,
            'regime': 'Mean-reversion',
            'entry_bar': 0,
            'bars_held': 0,
            'tp1_hit': False,
            'tp1_exit_price': None,
            'tp1_exit_time': None,
            'tp1_pnl': None,
            'runner_size': 0.4,  # 40% of position for runner
            'remaining_size': 1.0,  # Start with 100% position size
            'trailing_active': False,
            'high_since_entry': row['high'] if direction == 1 else row['low'],
            'low_since_entry': row['low'] if direction == 1 else row['high'],
            'mfe_history': [],  # Track MFE at each bar
            'mfa_history': [],  # Track MFA at each bar
            'velocity_check': False,  # Fast run check
            'stall_check': False,  # Stall filter check
            'early_adverse': False,  # Early adverse management
            'snap_lock': False,  # Snap-lock for strong extension
            'velocity_trail_adjustment': False,  # Velocity-based trail adjustment
            'scratch_reason': None,
            'entry_equity': self.current_equity
        }
        
        # Set mean reversion TP/SL parameters
        self.set_regime_parameters(position, row)
        
        return position
    
    def set_regime_parameters(self, position, row):
        """Set mean reversion TP/SL parameters"""
        entry_price = row['close']
        
        if position['direction'] == 1:  # MeanRev_Long
            position['tp1'] = entry_price * (1 + self.MEAN_REV_TP1_LONG)
            position['tp2'] = entry_price * (1 + self.MEAN_REV_TP2_LONG)
            position['stop_loss'] = entry_price * (1 - self.MEAN_REV_SL_LONG)
            position['trail_distance'] = 0.0020  # 0.20%
        else:  # MeanRev_Short
            position['tp1'] = entry_price * (1 - self.MEAN_REV_TP1_SHORT)
            position['tp2'] = entry_price * (1 - self.MEAN_REV_TP2_SHORT)
            position['stop_loss'] = entry_price * (1 + self.MEAN_REV_SL_SHORT)
            position['trail_distance'] = 0.0020  # 0.20%
    
    def update_position_tracking(self, position, row):
        """Update position tracking variables"""
        current_high = row['high']
        current_low = row['low']
        
        if position['direction'] == 1:  # Long
            position['high_since_entry'] = max(position['high_since_entry'], current_high)
            position['low_since_entry'] = min(position['low_since_entry'], current_low)
            
            # Calculate current MFE and MFA
            mfe = (position['high_since_entry'] - position['entry_price']) / position['entry_price']
            mfa = (position['low_since_entry'] - position['entry_price']) / position['entry_price']
        else:  # Short
            position['high_since_entry'] = max(position['high_since_entry'], current_high)
            position['low_since_entry'] = min(position['low_since_entry'], current_low)
            
            # Calculate current MFE and MFA
            mfe = (position['entry_price'] - position['low_since_entry']) / position['entry_price']
            mfa = (position['entry_price'] - position['high_since_entry']) / position['entry_price']
        
        position['mfe_history'].append(mfe)
        position['mfa_history'].append(mfa)
    
    def bar2_checks(self, position, row):
        """Bar 2 checks: velocity and early adverse management"""
        if len(position['mfe_history']) >= 2:
            # Fast run check (impulse)
            mfe_bar2 = position['mfe_history'][1]
            velocity_bar2 = mfe_bar2 / 2
            if velocity_bar2 >= self.VELOCITY_THRESHOLD:
                position['velocity_check'] = True
            
            # Early adverse management
            if len(position['mfa_history']) >= 2:
                mfa_bar2 = position['mfa_history'][1]
                if mfa_bar2 <= -0.0050 and not position['tp1_hit']:  # -0.50%
                    # Tighten stop loss
                    new_stop_loss = position['entry_price'] * (1 + 0.0050 if position['direction'] == -1 else 1 - 0.0050)
                    position['stop_loss'] = new_stop_loss
                    position['early_adverse'] = True
    
    def bar3_checks(self, position, row):
        """Bar 3 checks: stall filter and volume checks - EXECUTES stall scratch"""
        if len(position['mfe_history']) >= 3:
            mfe_bar3 = position['mfe_history'][2]
            if mfe_bar3 < self.STALL_FILTER_THRESHOLD:
                # Check volume stall
                if row['zvol'] < 1.0:
                    position['stall_check'] = True
                    position['scratch_reason'] = 'Volume stall at bar 3'
    
    def bar4_checks(self, position, row):
        """Bar 4 checks: snap-lock for strong extension and velocity-based trail adjustment"""
        if position['tp1_hit'] and len(position['mfe_history']) >= 4:
            mfe_bar4 = position['mfe_history'][3]
            if mfe_bar4 >= self.SNAP_LOCK_THRESHOLD:
                # Snap-lock: raise trail by +0.05%
                position['trail_distance'] += 0.0005
                position['snap_lock'] = True
        
        # Velocity-based trail tightening rule
        if position['tp1_hit'] and len(position['mfe_history']) >= 4:
            # If TP1 hit on bar 3-4, tighten trail to 0.15%
            if position['bars_held'] >= 3 and position['bars_held'] <= 4:
                if position['trail_distance'] > 0.0015:  # Only tighten if not already tight
                    position['trail_distance'] = 0.0015
                    position['velocity_trail_adjustment'] = True
    
    def check_exit_conditions(self, row, position):
        """Check exit conditions with 5-Bar Exit Engine logic"""
        # Check 5-bar hard time stop first
        if position['bars_held'] >= 5:
            return '5-Bar Time Stop'
        
        # Bar-specific checks
        if position['bars_held'] == 2:
            self.bar2_checks(position, row)
        elif position['bars_held'] == 3:
            self.bar3_checks(position, row)
        elif position['bars_held'] == 4:
            self.bar4_checks(position, row)
        
        # Check stall filter (Bar 3) - HIGH PRIORITY
        if position['stall_check']:
            return 'Stall Filter (Bar 3)'
        
        # Check stop loss
        if self.check_stop_loss(row, position):
            return 'Stop Loss'
        
        # Check TP1 (scale-out 60% of position)
        if not position['tp1_hit'] and self.check_tp1(row, position):
            return 'TP1 Hit (60% scale-out)'
        
        # Check TP2 (runner 40% of position)
        if position['tp1_hit'] and self.check_tp2(row, position):
            return 'TP2 Hit (Runner)'
        
        # Check trailing stop after TP1
        if position['tp1_hit'] and self.check_trailing_stop(row, position):
            return 'Trailing Stop'
        
        # Check catastrophic stop loss
        if self.check_catastrophic_stop(position):
            return 'Catastrophic Stop Loss'
        
        return None
    
    def check_stop_loss(self, row, position):
        """Check if stop loss is hit"""
        if position['direction'] == 1:  # Long
            return row['low'] <= position['stop_loss']
        else:  # Short
            return row['high'] >= position['stop_loss']
    
    def check_tp1(self, row, position):
        """Check if TP1 is hit"""
        if position['direction'] == 1:  # Long
            return row['high'] >= position['tp1']
        else:  # Short
            return row['low'] <= position['tp1']
    
    def check_tp2(self, row, position):
        """Check if TP2 is hit"""
        if position['direction'] == 1:  # Long
            return row['high'] >= position['tp2']
        else:  # Short
            return row['low'] <= position['tp2']
    
    def check_trailing_stop(self, row, position):
        """Check trailing stop after TP1"""
        if not position['tp1_hit']:
            return False
        
        # Activate trailing stop if not already active
        if not position['trailing_active']:
            position['trailing_active'] = True
            if position['direction'] == 1:  # Long
                position['trail_stop'] = position['high_since_entry'] * (1 - position['trail_distance'])
            else:  # Short
                position['trail_stop'] = position['low_since_entry'] * (1 + position['trail_distance'])
        
        # Update trailing stop
        if position['direction'] == 1:  # Long
            new_trail_stop = position['high_since_entry'] * (1 - position['trail_distance'])
            position['trail_stop'] = max(position['trail_stop'], new_trail_stop)
            
            # Check if trailing stop hit
            return row['low'] <= position['trail_stop']
        else:  # Short
            new_trail_stop = position['low_since_entry'] * (1 + position['trail_distance'])
            position['trail_stop'] = min(position['trail_stop'], new_trail_stop)
            
            # Check if trailing stop hit
            return row['high'] >= position['trail_stop']
    
    def check_catastrophic_stop(self, position):
        """Check catastrophic stop loss"""
        if len(position['mfa_history']) > 0:
            current_mfa = position['mfa_history'][-1]
            if current_mfa <= -self.CATASTROPHIC_STOP_THRESHOLD:
                return True
        return False
    
    def get_exit_price(self, row, position, exit_reason):
        """Get exit price based on exit reason"""
        if exit_reason == 'Stop Loss':
            return position['stop_loss']
        elif exit_reason == 'TP1 Hit (60% scale-out)':
            return position['tp1']
        elif exit_reason == 'TP2 Hit (Runner)':
            return position['tp2']
        elif exit_reason == 'Trailing Stop':
            return position['trail_stop']
        elif exit_reason == '5-Bar Time Stop':
            return row['close']  # Exit at current market price
        elif exit_reason == 'Stall Filter (Bar 3)':
            return row['close']  # Exit at current market price
        elif exit_reason == 'Catastrophic Stop Loss':
            return row['close']  # Exit at current market price
        else:
            return row['close']
    
    def execute_exit(self, exit_reason, row):
        """Execute exit based on exit reason"""
        if not self.position:
            return
        
        position = self.position
        exit_price = self.get_exit_price(row, position, exit_reason)
        
        # Calculate PnL based on exit reason
        if exit_reason == 'TP1 Hit (60% scale-out)':
            # Scale out 60% of position
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * position['direction']
            pnl_pct -= self.fee_per_trade  # Entry fee
            pnl_pct -= self.fee_per_trade * 0.6  # 60% exit fee
            
            # Update position
            position['tp1_hit'] = True
            position['tp1_exit_price'] = exit_price
            position['tp1_exit_time'] = row['timestamp']
            position['tp1_pnl'] = pnl_pct
            position['remaining_size'] = 0.4  # 40% remaining
            
            # Update equity
            self.current_equity *= (1 + pnl_pct * 0.6)  # 60% of position
            
            print(f"💰 TP1 Hit: 60% scale-out at {exit_price:.6f} | PnL: {pnl_pct*100:.3f}% | Equity: ${self.current_equity:.2f}")
            
        elif exit_reason == 'TP2 Hit (Runner)':
            # Exit remaining 40% of position
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * position['direction']
            pnl_pct -= self.fee_per_trade * 0.4  # 40% exit fee
            
            # Update equity
            self.current_equity *= (1 + pnl_pct * 0.4)  # 40% of position
            
            # Close position
            self.close_position(exit_reason, exit_price, row['timestamp'])
            
        else:
            # Full position exit
            pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * position['direction']
            pnl_pct -= self.round_trip_fee  # Full round trip fee
            
            # Update equity
            self.current_equity *= (1 + pnl_pct)
            
            # Close position
            self.close_position(exit_reason, exit_price, row['timestamp'])
    
    def close_position(self, exit_reason, exit_price, exit_time):
        """Close position and record trade"""
        if not self.position:
            return
        
        position = self.position
        
        # Calculate final PnL
        if exit_reason == 'TP1 Hit (60% scale-out)':
            # Already handled in execute_exit
            return
        
        pnl_pct = (exit_price - position['entry_price']) / position['entry_price'] * position['direction']
        pnl_pct -= self.round_trip_fee
        
        # Record trade
        trade = {
            'entry_time': position['entry_time'],
            'exit_time': exit_time,
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'direction': position['direction'],
            'signal_type': position['signal_type'],
            'exit_reason': exit_reason,
            'bars_held': position['bars_held'],
            'pnl_pct': pnl_pct,
            'pnl_dollar': pnl_pct * position['entry_equity'],
            'entry_equity': position['entry_equity'],
            'exit_equity': self.current_equity,
            'mfe': max(position['mfe_history']) if position['mfe_history'] else 0,
            'mfa': min(position['mfa_history']) if position['mfa_history'] else 0,
            'tp1_hit': position['tp1_hit'],
            'tp1_pnl': position['tp1_pnl'],
            'velocity_check': position['velocity_check'],
            'stall_check': position['stall_check'],
            'early_adverse': position['early_adverse'],
            'snap_lock': position['snap_lock'],
            'scratch_reason': position['scratch_reason']
        }
        
        self.trades.append(trade)
        
        # Update daily tracking
        self.update_daily_tracking(exit_reason, pnl_pct)
        
        print(f"🚪 EXIT: {exit_reason} at {exit_price:.6f} | "
              f"Bars: {position['bars_held']} | "
              f"PnL: {pnl_pct*100:.3f}% | "
              f"Equity: ${self.current_equity:.2f}")
        
        # Clear position
        self.position = None
    
    def update_daily_tracking(self, exit_reason, pnl_pct):
        """Update daily tracking for guardrails"""
        self.daily_pnl += pnl_pct
        if exit_reason == '5-Bar Time Stop':
            self.daily_time_stops += 1
    
    def run_backtest(self, df):
        """Run the backtest"""
        print(f"🚀 Starting backtest on {len(df)} candles...")
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Process each candle
        for i, (idx, row) in enumerate(df.iterrows()):
            if i % 1000 == 0:
                print(f"📊 Processing candle {i}/{len(df)}")
            
            # Skip if insufficient data for indicators
            if pd.isna(row['vwap']) or pd.isna(row['atr']):
                continue
            
            # Check for new signal if not in position
            if self.position is None:
                # Check daily guardrails
                if self.should_skip_signal(row['timestamp']):
                    continue
                
                # Check for entry signal
                signal = self.check_entry_signal(row)
                
                if signal:
                    signal_type, direction = signal
                    print(f"🎯 ENTRY: {signal_type} at {row['close']:.6f}")
                    
                    # Create position
                    self.position = self.create_position(row, signal_type, direction)
                    self.last_signal_time = row['timestamp']
            
            else:
                # Update position tracking
                self.update_position_tracking(self.position, row)
                
                # Increment bars held
                self.position['bars_held'] += 1
                
                # Check exit conditions
                exit_reason = self.check_exit_conditions(row, self.position)
                
                if exit_reason:
                    self.execute_exit(exit_reason, row)
            
            # Update equity curve
            self.equity_curve.append(self.current_equity)
        
        print(f"✅ Backtest completed!")
        print(f"📊 Total trades: {len(self.trades)}")
        print(f"💰 Final equity: ${self.current_equity:.2f}")
        print(f"📈 Total return: {((self.current_equity / self.initial_capital) - 1) * 100:.2f}%")
    
    def analyze_results(self):
        """Analyze backtest results"""
        if not self.trades:
            print("❌ No trades to analyze")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        print(f"\n{'='*60}")
        print(f"📊 BACKTEST RESULTS ANALYSIS")
        print(f"{'='*60}")
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
        win_rate = winning_trades / total_trades * 100
        
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() * 100
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() * 100
        profit_factor = abs(trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum() / trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].sum())
        
        total_return = ((self.current_equity / self.initial_capital) - 1) * 100
        max_equity = max(self.equity_curve)
        max_drawdown = ((max_equity - min(self.equity_curve)) / max_equity) * 100
        
        print(f"💰 PERFORMANCE METRICS:")
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Final Equity: ${self.current_equity:,.2f}")
        print(f"   Max Drawdown: {max_drawdown:.2f}%")
        print(f"   Profit Factor: {profit_factor:.2f}")
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Avg Win: {avg_win:.3f}%")
        print(f"   Avg Loss: {avg_loss:.3f}%")
        print(f"   Win/Loss Ratio: {abs(avg_win/avg_loss):.2f}")
        
        # Exit reason analysis
        print(f"\n🚪 EXIT REASON ANALYSIS:")
        exit_counts = trades_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            pct = count / total_trades * 100
            avg_pnl = trades_df[trades_df['exit_reason'] == reason]['pnl_pct'].mean() * 100
            print(f"   {reason}: {count} ({pct:.1f}%) | Avg PnL: {avg_pnl:.3f}%")
        
        # Signal type analysis
        print(f"\n🎯 SIGNAL TYPE ANALYSIS:")
        signal_counts = trades_df['signal_type'].value_counts()
        for signal, count in signal_counts.items():
            pct = count / total_trades * 100
            avg_pnl = trades_df[trades_df['signal_type'] == signal]['pnl_pct'].mean() * 100
            print(f"   {signal}: {count} ({pct:.1f}%) | Avg PnL: {avg_pnl:.3f}%")
        
        # 5-Bar Exit Engine analysis
        print(f"\n⏰ 5-BAR EXIT ENGINE ANALYSIS:")
        tp1_hits = trades_df['tp1_hit'].sum()
        tp1_rate = tp1_hits / total_trades * 100
        print(f"   TP1 Hit Rate: {tp1_rate:.1f}% ({tp1_hits}/{total_trades})")
        
        if tp1_hits > 0:
            tp1_trades = trades_df[trades_df['tp1_hit'] == True]
            avg_tp1_pnl = tp1_trades['tp1_pnl'].mean() * 100
            print(f"   Avg TP1 PnL: {avg_tp1_pnl:.3f}%")
        
        # Bar holding analysis
        avg_bars_held = trades_df['bars_held'].mean()
        print(f"   Avg Bars Held: {avg_bars_held:.1f}")
        
        # MFE/MFA analysis
        avg_mfe = trades_df['mfe'].mean() * 100
        avg_mfa = trades_df['mfa'].mean() * 100
        print(f"   Avg MFE: {avg_mfe:.3f}%")
        print(f"   Avg MFA: {avg_mfa:.3f}%")
        
        return trades_df
    
    def plot_results(self, trades_df):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Equity curve
        axes[0, 0].plot(self.equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Equity ($)')
        axes[0, 0].grid(True)
        
        # PnL distribution
        axes[0, 1].hist(trades_df['pnl_pct'] * 100, bins=30, alpha=0.7)
        axes[0, 1].set_title('PnL Distribution')
        axes[0, 1].set_xlabel('PnL (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Exit reasons
        exit_counts = trades_df['exit_reason'].value_counts()
        axes[1, 0].pie(exit_counts.values, labels=exit_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Exit Reasons')
        
        # Bars held distribution
        axes[1, 1].hist(trades_df['bars_held'], bins=range(1, 8), alpha=0.7)
        axes[1, 1].set_title('Bars Held Distribution')
        axes[1, 1].set_xlabel('Bars Held')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('edge5_ravf_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Results plotted and saved as 'edge5_ravf_backtest_results.png'")

def load_data(file_path):
    """Load data from CSV file"""
    print(f"📁 Loading data from {file_path}...")
    
    try:
        # Check if this is the live data format (datadoge.csv)
        if file_path.endswith("datadoge.csv"):
            # Live data format: Timestamp,DateTime,Open,High,Low,Close,Volume
            df = pd.read_csv(file_path)
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'Timestamp': 'timestamp',
                'DateTime': 'datetime',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Convert timestamp from milliseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
        else:
            # Historical data format: timestamp, open, high, low, close, volume, vwap, trades, buy_volume, buy_value
            df = pd.read_csv(file_path, header=None)
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trades', 'buy_volume', 'buy_value']
            
            # Convert timestamp from nanoseconds to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        
        # Ensure we have the required columns
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return None
        
        print(f"✅ Data loaded: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def main():
    """Main function to run backtest on all available months"""
    # Check for --live flag
    use_live_data = "--live" in sys.argv
    
    # Initialize backtester
    backtester = Edge5RAVFBacktester(initial_capital=10000, fee_per_trade=0.0003)
    
    if use_live_data:
        # Use datadoge.csv for live data testing
        live_data_file = "datadoge.csv"
        if not os.path.exists(live_data_file):
            print(f"❌ Live data file not found: {live_data_file}")
            return
        
        print(f"🔄 Running backtest with live data: {live_data_file}")
        data_files = [live_data_file]
    else:
        # Get all available data files from Backtest/5m
        data_dir = "Backtest/5m"
        data_files = []
        
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.startswith("DOGEUSDT-5m-2025-") and file.endswith(".csv"):
                    data_files.append(os.path.join(data_dir, file))
        
        if not data_files:
            print("❌ No data files found in Backtest/5m/")
            return
        
        # Sort files by month
        data_files.sort()
        print(f"📁 Found {len(data_files)} data files: {[os.path.basename(f) for f in data_files]}")
    
    # Process all files
    all_trades = []
    total_candles = 0
    
    for i, data_file in enumerate(data_files):
        print(f"\n{'='*60}")
        print(f"📊 Processing {os.path.basename(data_file)} ({i+1}/{len(data_files)})")
        print(f"{'='*60}")
        
        # Load data
        df = load_data(data_file)
        
        if df is None:
            print(f"❌ Failed to load {data_file}")
            continue
        
        total_candles += len(df)
        
        # Run backtest on this month
        month_trades = backtester.run_backtest(df)
        
        # Collect trades for this month
        if backtester.trades:
            all_trades.extend(backtester.trades)
            print(f"✅ Month completed: {len(backtester.trades)} trades")
        else:
            print("⚠️ No trades generated for this month")
        
        # Reset for next month (keep equity curve)
        backtester.trades = []
        backtester.position = None
        backtester.daily_pnl = 0
        backtester.daily_time_stops = 0
        backtester.last_signal_time = None
        backtester.current_date = None
    
    # Restore all trades to backtester for final analysis
    backtester.trades = all_trades
    
    print(f"\n{'='*60}")
    print(f"📊 FINAL RESULTS - ALL MONTHS COMBINED")
    print(f"{'='*60}")
    print(f"📈 Total candles processed: {total_candles:,}")
    print(f"📊 Total trades: {len(all_trades)}")
    print(f"💰 Final equity: ${backtester.current_equity:,.2f}")
    print(f"📈 Total return: {((backtester.current_equity / backtester.initial_capital) - 1) * 100:.2f}%")
    
    # Analyze results
    if all_trades:
        trades_df = backtester.analyze_results()
        
        # Plot results
        if trades_df is not None and len(trades_df) > 0:
            backtester.plot_results(trades_df)
            
            # Save trades to CSV
            trades_df.to_csv('edge5_ravf_trades_all_months.csv', index=False)
            print("💾 Trades saved to 'edge5_ravf_trades_all_months.csv'")
            
            # Create simple entry/exit prices CSV
            simple_trades = trades_df[['entry_time', 'exit_time', 'entry_price', 'exit_price', 'direction', 'signal_type', 'exit_reason', 'pnl_pct']].copy()
            simple_trades['direction_text'] = simple_trades['direction'].map({1: 'LONG', -1: 'SHORT'})
            simple_trades['pnl_pct'] = simple_trades['pnl_pct'] * 100  # Convert to percentage
            simple_trades = simple_trades.rename(columns={
                'entry_time': 'Entry Time',
                'exit_time': 'Exit Time', 
                'entry_price': 'Entry Price',
                'exit_price': 'Exit Price',
                'direction_text': 'Direction',
                'signal_type': 'Signal Type',
                'exit_reason': 'Exit Reason',
                'pnl_pct': 'PnL %'
            })
            simple_trades.to_csv('edge5_ravf_simple_trades_all_months.csv', index=False)
            print("💾 Simple trades (entry/exit prices) saved to 'edge5_ravf_simple_trades_all_months.csv'")
    else:
        print("❌ No trades generated across all months")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h"]:
        print("EDGE-5 RAVF Strategy Backtester")
        print("===============================")
        print("Usage:")
        print("  python edge5_ravf_backtester.py          # Run with historical data (Backtest/5m/)")
        print("  python edge5_ravf_backtester.py --live   # Run with live data (datadoge.csv)")
        print("  python edge5_ravf_backtester.py --help   # Show this help")
        sys.exit(0)
    
    main()
