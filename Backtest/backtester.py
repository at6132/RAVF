import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Edge5RAVFBacktester:
    """
    EDGE-5 RAVF Backtester - MEAN REVERSION ONLY
    
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
    
    def __init__(self, fee_per_trade=0.0003):
        """
        Initialize backtester
        
        Args:
            fee_per_trade: Fee per trade (0.03% = 0.0003)
        """
        self.fee_per_trade = fee_per_trade
        self.round_trip_fee = fee_per_trade * 2  # 0.06% round trip
        
        # Backtest results
        self.trades = []
        self.equity_curve = []
        
        # Daily guardrails tracking
        self.daily_pnl = 0
        self.daily_time_stops = 0
        self.last_signal_time = None
        
        print(f"Mean Reversion Strategy Only")
        print(f"Round-trip fees: {self.round_trip_fee*100:.2f}%")
    
    def run_backtest(self, data_dir="5m"):
        """Run backtest on 5-minute data"""
        import os
        
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} not found!")
            return
        
        # Get all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
        csv_files.sort()
        
        print(f"\nRunning backtest on {len(csv_files)} files...")
        
        all_trades = []
        current_equity = 10000  # Starting capital
        
        # Initialize equity curve as instance variable
        self.equity_curve = [current_equity]
        
        for csv_file in csv_files:
            print(f"Processing {csv_file}...")
            file_path = os.path.join(data_dir, csv_file)
            
            # Read data
            df = pd.read_csv(file_path, header=None, names=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote'
            ])
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
            df['date'] = df['timestamp'].dt.date
            
            # Run strategy to get signals
            df_with_signals = self._run_strategy(df)
            
            # Execute trades
            trades, equity = self._execute_trades(df_with_signals, current_equity)
            
            if trades:
                all_trades.extend(trades)
                current_equity = equity
                print(f"Processed {len(trades)} trades, current equity: ${current_equity:,.2f}")
        
        self.trades = all_trades
        
        # Add final equity if not already added
        if self.equity_curve[-1] != current_equity:
            self.equity_curve.append(current_equity)
        
        print(f"Equity curve length: {len(self.equity_curve)}")
        print(f"Final equity: ${current_equity:,.2f}")
        print(f"Total trades: {len(all_trades)}")
        
        return self._generate_backtest_report()
    
    def _run_strategy(self, df):
        """Run the EDGE-5 RAVF strategy - MEAN REVERSION ONLY"""
        # Calculate basic indicators
        df['returns'] = np.log(df['close'] / df['close'].shift(1))
        df['rv'] = df['returns'].rolling(window=48).std()
        df['skew'] = df['returns'].rolling(window=48).skew()
        df['kurt'] = df['returns'].rolling(window=48).kurt()
        df['atr'] = self._calculate_atr(df)
        df['clv'] = self._calculate_clv(df)
        df['vwap'] = self._calculate_vwap(df)
        df['zvol'] = self._calculate_relative_volume(df)
        
        # Detect regime - ONLY MEAN REVERSION
        df = self._detect_regime(df)
        
        # Generate signals - ONLY MEAN REVERSION
        df = self._generate_signals(df)
        
        return df
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                np.abs(df['high'] - df['close'].shift(1)),
                np.abs(df['low'] - df['close'].shift(1))
            )
        )
        return df['tr'].rolling(window=window).mean()
    
    def _calculate_clv(self, df):
        """Calculate Close Location Value"""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        return clv.clip(-1, 1)
    
    def _calculate_vwap(self, df, window=48):
        """Calculate rolling VWAP to avoid look-ahead bias"""
        volume_price = df['close'] * df['volume']
        rolling_vwap = volume_price.rolling(window=window).sum() / df['volume'].rolling(window=window).sum()
        return rolling_vwap
    
    def _calculate_relative_volume(self, df, lookback=48):
        """Calculate relative volume"""
        return df['volume'] / df['volume'].rolling(window=lookback).mean()
    
    def _detect_regime(self, df):
        """Detect market regime - ONLY MEAN REVERSION"""
        df['regime'] = 'Mean-reversion'  # Force all to mean reversion
        
        # Calculate percentiles for RV over rolling 60-day lookback
        df['rv_percentile'] = df['rv'].rolling(window=288).rank(pct=True)  # 288 = 24 hours
        
        for i in range(len(df)):
            if pd.isna(df.iloc[i]['rv']) or pd.isna(df.iloc[i]['skew']) or pd.isna(df.iloc[i]['kurt']):
                continue
                
            # Force mean reversion regime
            df.loc[df.index[i], 'regime'] = 'Mean-reversion'
        
        return df
    
    def _generate_signals(self, df):
        """Generate entry signals - MEAN REVERSION ONLY"""
        df['signal'] = 0  # 0: no signal, 1: long, -1: short
        df['signal_type'] = ''
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        
        for i in range(1, len(df)):
            if i < 14:  # Need enough data for ATR
                continue
                
            current = df.iloc[i]
            prev = df.iloc[i-1]
            
            # MEAN-REVERSION AVWAP SNAPBACK LONG
            if (current['regime'] == 'Mean-reversion' and
                current['close'] <= current['vwap'] - 0.6 * current['atr'] and
                current['clv'] <= -0.4 and
                current['zvol'] >= 1.2):
                
                df.loc[df.index[i], 'signal'] = 1
                df.loc[df.index[i], 'signal_type'] = 'MeanRev_Long'
                df.loc[df.index[i], 'entry_price'] = current['close']
                df.loc[df.index[i], 'stop_loss'] = current['vwap'] - 1.0 * current['atr']
            
            # MEAN-REVERSION AVWAP SNAPBACK SHORT
            elif (current['regime'] == 'Mean-reversion' and
                  current['close'] >= current['vwap'] + 0.6 * current['atr'] and
                  current['clv'] >= 0.4 and
                  current['zvol'] >= 1.2):
                
                df.loc[df.index[i], 'signal'] = -1
                df.loc[df.index[i], 'signal_type'] = 'MeanRev_Short'
                df.loc[df.index[i], 'entry_price'] = current['close']
                df.loc[df.index[i], 'stop_loss'] = current['vwap'] + 1.0 * current['atr']
        
        return df
    
    def _should_skip_signal(self, current):
        """Check daily guardrails"""
        # Reset daily tracking if new day
        if hasattr(self, 'current_date') and self.current_date != current['timestamp'].date():
            self.daily_pnl = 0
            self.daily_time_stops = 0
        
        self.current_date = current['timestamp'].date()
        
        # Check if we should stop taking signals
        if self.daily_pnl <= -1.5 * 0.0085 or self.daily_time_stops >= 3:  # -1.5R or 3 time-stops
            # Check if 6 hours have passed since last signal
            if self.last_signal_time and (current['timestamp'] - self.last_signal_time).total_seconds() < 6 * 3600:
                return True
        
        return False
    
    def _execute_trades(self, df, starting_equity):
        """Execute trades with 5-Bar Exit Engine"""
        trades = []
        current_equity = starting_equity
        position = None
        
        for i in range(len(df)):
            current = df.iloc[i]
            
            # Check for new signal
            if current['signal'] != 0 and position is None:
                # Check daily guardrails
                if self._should_skip_signal(current):
                    continue
                    
                # Open new position
                position = self._create_position(current, i)
                position['entry_equity'] = current_equity
                position['remaining_size'] = 1.0  # Start with 100% position size
                self.last_signal_time = current['timestamp']
            
            # Check for exit conditions if in position
            if position is not None:
                # Update position tracking
                self._update_position_tracking(position, current)
                
                # Increment bars held
                position['bars_held'] += 1
                
                # Check exit conditions
                exit_reason = self._check_exit_conditions(current, position, i)
                
                if exit_reason:
                    exit_price = self._get_exit_price(current, position, exit_reason)
                    
                    # Handle scale-out logic
                    if exit_reason == 'TP1 Hit (60% scale-out)':
                        # Calculate P&L for 60% of position
                        tp1_pnl = self._calculate_pnl(position, position['tp1']) * 0.6
                        total_fees_60 = self.fee_per_trade * 0.6 * 2  # Entry + exit fees
                        net_tp1_pnl = tp1_pnl - total_fees_60
                        
                        # Convert percentage P&L to dollar amount using current equity
                        net_tp1_pnl_dollars = net_tp1_pnl * current_equity
                        current_equity += net_tp1_pnl_dollars
                        
                        # Update equity curve after partial exit
                        self.equity_curve.append(current_equity)
                        
                        # Update daily tracking
                        self._update_daily_tracking(exit_reason, net_tp1_pnl_dollars)
                        
                        # Record partial exit trade
                        trade = self._record_trade(position, current, exit_reason, position['tp1'], 
                                                 tp1_pnl, total_fees_60, net_tp1_pnl, current_equity, 
                                                 position_size=0.6, is_partial=True)
                        trades.append(trade)
                        
                        # Continue with 40% runner
                        position['remaining_size'] = 0.4
                        position['tp1_hit'] = True
                        position['tp1_exit_price'] = position['tp1']
                        position['tp1_exit_time'] = current['timestamp']
                        position['tp1_pnl'] = tp1_pnl
                        
                    elif position['tp1_hit'] and exit_reason in ['TP2 Hit (Runner)', 'Trailing Stop']:
                        # Calculate final 40% P&L - NO ENTRY FEES for runner!
                        raw_runner_pnl = self._calculate_pnl(position, exit_price) * 0.4
                        exit_fee_pct = self.fee_per_trade * 0.4  # Only exit fee on 40%
                        net_runner_pnl = raw_runner_pnl - exit_fee_pct
                        
                        # Convert percentage P&L to dollar amount using current equity
                        net_runner_pnl_dollars = net_runner_pnl * current_equity
                        current_equity += net_runner_pnl_dollars
                        
                        # Update equity curve after final exit
                        self.equity_curve.append(current_equity)
                        
                        # Update daily tracking
                        self._update_daily_tracking(exit_reason, net_runner_pnl_dollars)
                        
                        # Record final exit trade
                        trade = self._record_trade(position, current, exit_reason, exit_price, 
                                                 raw_runner_pnl, exit_fee_pct, net_runner_pnl, current_equity,
                                                 position_size=0.4, is_partial=True)
                        trades.append(trade)
                        
                        # Now close position completely
                        position = None
                        
                    else:
                        # Full position exit (stop loss, time stop, etc.)
                        # Calculate full position P&L
                        pnl = self._calculate_pnl(position, exit_price)
                        total_fees = self.fee_per_trade * 2  # Entry + exit fees
                        net_pnl = pnl - total_fees
                        
                        # Convert percentage P&L to dollar amount using current equity
                        net_pnl_dollars = net_pnl * current_equity
                        current_equity += net_pnl_dollars
                        
                        # Update equity curve after full exit
                        self.equity_curve.append(current_equity)
                        
                        # Update daily tracking
                        self._update_daily_tracking(exit_reason, net_pnl_dollars)
                        
                        # Record full exit trade
                        trade = self._record_trade(position, current, exit_reason, exit_price, 
                                                 pnl, total_fees, net_pnl, current_equity,
                                                 position_size=1.0, is_partial=False)
                        trades.append(trade)
                        
                        position = None
        
        return trades, current_equity
    
    def _create_position(self, current, i):
        """Create new position with mean reversion parameters"""
        position = {
            'entry_time': current['timestamp'],
            'entry_price': current['entry_price'],
            'signal_type': current['signal_type'],
            'direction': current['signal'],
            'regime': current['regime'],
            'entry_bar': i,
            'bars_held': 0,
            'tp1_hit': False,
            'tp1_exit_price': None,
            'tp1_exit_time': None,
            'tp1_pnl': None,
            'runner_size': 0.4,  # 40% of position for runner
            'remaining_size': 1.0,  # Start with 100% position size
            'trailing_active': False,
            'high_since_entry': current['high'] if current['signal'] == 1 else current['low'],
            'low_since_entry': current['low'] if current['signal'] == 1 else current['high'],
            'mfe_history': [],  # Track MFE at each bar
            'mfa_history': [],  # Track MFA at each bar
            'velocity_check': False,  # Fast run check
            'stall_check': False,  # Stall filter check
            'early_adverse': False,  # Early adverse management
            'snap_lock': False,  # Snap-lock for strong extension
            'velocity_trail_adjustment': False,  # Velocity-based trail adjustment
            'scratch_reason': None,
            'entry_equity': None  # Will be set when position is created
        }
        
        # Set mean reversion TP/SL parameters
        self._set_regime_parameters(position, current)
        
        return position
    
    def _set_regime_parameters(self, position, current):
        """Set mean reversion TP/SL parameters"""
        if position['direction'] == 1:  # MeanRev_Long
            position['tp1'] = current['close'] * 1.0050  # +0.50%
            position['tp2'] = current['close'] * 1.0095  # +0.95%
            position['stop_loss'] = current['close'] * 0.9920  # -0.80%
            position['trail_distance'] = 0.0020  # 0.20%
        else:  # MeanRev_Short
            position['tp1'] = current['close'] * 0.9945  # -0.55%
            position['tp2'] = current['close'] * 0.9900  # -1.00%
            position['stop_loss'] = current['close'] * 1.0080  # +0.80%
            position['trail_distance'] = 0.0020  # 0.20%
    
    def _update_position_tracking(self, position, current):
        """Update position tracking variables"""
        if position['direction'] == 1:  # Long
            position['high_since_entry'] = max(position['high_since_entry'], current['high'])
            position['low_since_entry'] = min(position['low_since_entry'], current['low'])
            
            # Calculate current MFE and MFA
            mfe = (position['high_since_entry'] - position['entry_price']) / position['entry_price']
            mfa = (position['low_since_entry'] - position['entry_price']) / position['entry_price']
        else:  # Short
            position['high_since_entry'] = max(position['high_since_entry'], current['high'])
            position['low_since_entry'] = min(position['low_since_entry'], current['low'])
            
            # Calculate current MFE and MFA
            mfe = (position['entry_price'] - position['low_since_entry']) / position['entry_price']
            mfa = (position['entry_price'] - position['high_since_entry']) / position['entry_price']
        
        position['mfe_history'].append(mfe)
        position['mfa_history'].append(mfa)
    
    def _check_exit_conditions(self, current, position, current_bar):
        """Check exit conditions with 5-Bar Exit Engine logic"""
        # Check 5-bar hard time stop first
        if position['bars_held'] >= 5:
            return '5-Bar Time Stop'
        
        # Bar-specific checks
        if position['bars_held'] == 2:
            self._bar2_checks(position, current)
        elif position['bars_held'] == 3:
            self._bar3_checks(position, current)
        elif position['bars_held'] == 4:
            self._bar4_checks(position, current)
        
        # Check stall filter (Bar 3) - HIGH PRIORITY
        if position['stall_check']:
            return 'Stall Filter (Bar 3)'
        
        # Check stop loss
        if self._check_stop_loss(current, position):
            return 'Stop Loss'
        
        # Check TP1 (scale-out 60% of position)
        if not position['tp1_hit'] and self._check_tp1(current, position):
            return 'TP1 Hit (60% scale-out)'
        
        # Check TP2 (runner 40% of position)
        if position['tp1_hit'] and self._check_tp2(current, position):
            return 'TP2 Hit (Runner)'
        
        # Check trailing stop after TP1
        if position['tp1_hit'] and self._check_trailing_stop(current, position):
            return 'Trailing Stop'
        
        # Check catastrophic stop loss
        if self._check_catastrophic_stop(position):
            return 'Catastrophic Stop Loss'
        
        return None
    
    def _bar2_checks(self, position, current):
        """Bar 2 checks: velocity and early adverse management"""
        if len(position['mfe_history']) >= 2:
            # Fast run check (impulse)
            mfe_bar2 = position['mfe_history'][1]
            velocity_bar2 = mfe_bar2 / 2
            if velocity_bar2 >= 0.00225:  # 0.225%/bar
                position['velocity_check'] = True
            
            # Early adverse management
            if len(position['mfa_history']) >= 2:
                mfa_bar2 = position['mfa_history'][1]
                if mfa_bar2 <= -0.0050 and not position['tp1_hit']:  # -0.50%
                    # Tighten stop loss
                    position['stop_loss'] = position['entry_price'] * (1 + 0.0050 if position['direction'] == -1 else 1 - 0.0050)
                    position['early_adverse'] = True
    
    def _bar3_checks(self, position, current):
        """Bar 3 checks: stall filter and volume checks - EXECUTES stall scratch"""
        if len(position['mfe_history']) >= 3:
            mfe_bar3 = position['mfe_history'][2]
            if mfe_bar3 < 0.0025:  # < +0.25%
                # Check volume stall
                if hasattr(current, 'zvol') and current['zvol'] < 1.0:
                    position['stall_check'] = True
                    position['scratch_reason'] = 'Volume stall at bar 3'
    
    def _bar4_checks(self, position, current):
        """Bar 4 checks: snap-lock for strong extension and velocity-based trail adjustment"""
        if position['tp1_hit'] and len(position['mfe_history']) >= 4:
            mfe_bar4 = position['mfe_history'][3]
            if mfe_bar4 >= 0.0080:  # ≥ +0.80%
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
    
    def _check_stop_loss(self, current, position):
        """Check if stop loss is hit"""
        if position['direction'] == 1:  # Long
            return current['low'] <= position['stop_loss']
        else:  # Short
            return current['high'] >= position['stop_loss']
    
    def _check_tp1(self, current, position):
        """Check if TP1 is hit"""
        if position['direction'] == 1:  # Long
            return current['high'] >= position['tp1']
        else:  # Short
            return current['low'] <= position['tp1']
    
    def _check_tp2(self, current, position):
        """Check if TP2 is hit"""
        if position['direction'] == 1:  # Long
            return current['high'] >= position['tp2']
        else:  # Short
            return current['low'] <= position['tp2']
    
    def _check_trailing_stop(self, current, position):
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
            return current['low'] <= position['trail_stop']
        else:  # Short
            new_trail_stop = position['low_since_entry'] * (1 + position['trail_distance'])
            position['trail_stop'] = min(position['trail_stop'], new_trail_stop)
            
            # Check if trailing stop hit
            return current['high'] >= position['trail_stop']
    
    def _check_catastrophic_stop(self, position):
        """Check catastrophic stop loss"""
        if len(position['mfa_history']) > 0:
            current_mfa = position['mfa_history'][-1]
            if current_mfa <= -0.0090:  # ≤ -0.90%
                return True
        return False
    
    def _calculate_pnl(self, position, exit_price):
        """Calculate P&L for position"""
        if position['direction'] == 1:  # Long
            return (exit_price - position['entry_price']) / position['entry_price']
        else:  # Short
            return (position['entry_price'] - exit_price) / position['entry_price']
    
    def _get_exit_price(self, current, position, exit_reason):
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
            return current['close']  # Exit at current market price
        elif exit_reason == 'Stall Filter (Bar 3)':
            return current['close']  # Exit at current market price
        elif exit_reason == 'Catastrophic Stop Loss':
            return current['close']  # Exit at current market price
        else:
            return current['close']
    
    def _update_daily_tracking(self, exit_reason, net_pnl):
        """Update daily tracking for guardrails"""
        self.daily_pnl += net_pnl
        if exit_reason == '5-Bar Time Stop':
            self.daily_time_stops += 1
    
    def _record_trade(self, position, current, exit_reason, exit_price, pnl, total_fees, net_pnl, current_equity, position_size=1.0, is_partial=False):
        """Record trade details"""
        trade_record = {
            'entry_time': position['entry_time'],
            'exit_time': current['timestamp'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'signal_type': position['signal_type'],
            'direction': position['direction'],
            'regime': position['regime'],
            'exit_reason': exit_reason,
            'bars_held': position['bars_held'],
            'tp1_hit': position['tp1_hit'],
            'tp1_exit_price': position['tp1_exit_price'],
            'tp1_exit_time': position['tp1_exit_time'],
            'tp1_pnl': position['tp1_pnl'],
            'velocity_check': position['velocity_check'],
            'stall_check': position['stall_check'],
            'early_adverse': position['early_adverse'],
            'snap_lock': position['snap_lock'],
            'velocity_trail_adjustment': position['velocity_trail_adjustment'],
            'scratch_reason': position['scratch_reason'],
            'final_mfe': position['mfe_history'][-1] if position['mfe_history'] else None,
            'final_mfa': position['mfa_history'][-1] if position['mfa_history'] else None,
            'pnl': pnl,
            'fees': total_fees,
            'net_pnl': net_pnl,
            'entry_equity': position['entry_equity'],
            'exit_equity': current_equity,
            'position_size': position_size,
            'is_partial': is_partial,
            'remaining_size': position.get('remaining_size', 1.0)
        }
        
        return trade_record
    
    def _generate_backtest_report(self):
        """Generate comprehensive backtest report"""
        if not self.trades:
            print("No trades executed!")
            return
        
        trades_df = pd.DataFrame(self.trades)
        
        print("\n" + "=" * 80)
        print("EDGE-5 RAVF BACKTEST RESULTS - MEAN REVERSION ONLY")
        print("=" * 80)
        
        # Basic statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] < 0])
        win_rate = winning_trades / total_trades * 100
        
        print(f"\n📊 TRADE STATISTICS:")
        print(f"Total Trades: {total_trades:,}")
        print(f"Winning Trades: {winning_trades:,} ({win_rate:.1f}%)")
        print(f"Losing Trades: {losing_trades:,} ({100-win_rate:.1f}%)")
        
        # Performance metrics
        total_return = (self.equity_curve[-1] - self.equity_curve[0]) / self.equity_curve[0] * 100
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() * 100
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() * 100
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        print(f"\n💰 PERFORMANCE METRICS:")
        print(f"Starting Equity: ${self.equity_curve[0]:,.2f}")
        print(f"Ending Equity: ${self.equity_curve[-1]:,.2f}")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Average Win: {avg_win:.3f}%")
        print(f"Average Loss: {avg_loss:.3f}%")
        print(f"Profit Factor: {profit_factor:.2f}")
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()
        
        print(f"\n⚠️ RISK METRICS:")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # By signal type
        print(f"\n📈 PERFORMANCE BY SIGNAL TYPE:")
        for signal_type in trades_df['signal_type'].unique():
            type_trades = trades_df[trades_df['signal_type'] == signal_type]
            type_win_rate = len(type_trades[type_trades['net_pnl'] > 0]) / len(type_trades) * 100
            type_avg_pnl = type_trades['net_pnl'].mean() * 100
            print(f"{signal_type}: {len(type_trades)} trades, {type_win_rate:.1f}% win rate, {type_avg_pnl:.3f}% avg P&L")
        
        # By exit reason
        print(f"\n🚪 EXIT REASON ANALYSIS:")
        for exit_reason in trades_df['exit_reason'].unique():
            reason_trades = trades_df[trades_df['exit_reason'] == exit_reason]
            reason_win_rate = len(reason_trades[reason_trades['net_pnl'] > 0]) / len(reason_trades) * 100
            reason_avg_pnl = reason_trades['net_pnl'].mean() * 100
            print(f"{exit_reason}: {len(reason_trades)} trades, {reason_win_rate:.1f}% win rate, {reason_avg_pnl:.3f}% avg P&L")
        
        # 5-Bar Exit Engine specific analysis
        print(f"\n🎯 5-BAR EXIT ENGINE ANALYSIS:")
        
        # Scale-out analysis
        partial_trades = trades_df[trades_df['is_partial'] == True]
        full_trades = trades_df[trades_df['is_partial'] == False]
        
        if len(partial_trades) > 0:
            print(f"Partial exits (scale-outs): {len(partial_trades)} ({len(partial_trades)/len(trades_df)*100:.1f}%)")
            print(f"Full exits: {len(full_trades)} ({len(full_trades)/len(trades_df)*100:.1f}%)")
            
            # Analyze partial exits by position size
            tp1_exits = partial_trades[partial_trades['position_size'] == 0.6]
            runner_exits = partial_trades[partial_trades['position_size'] == 0.4]
            
            if len(tp1_exits) > 0:
                print(f"TP1 exits (60%): {len(tp1_exits)}, avg P&L: {tp1_exits['net_pnl'].mean()*100:.3f}%")
            if len(runner_exits) > 0:
                print(f"Runner exits (40%): {len(runner_exits)}, avg P&L: {runner_exits['net_pnl'].mean()*100:.3f}%")
        
        # TP1 analysis
        tp1_trades = trades_df[trades_df['tp1_hit'] == True]
        if len(tp1_trades) > 0:
            print(f"Trades that hit TP1: {len(tp1_trades)} ({len(tp1_trades)/len(trades_df)*100:.1f}%)")
            print(f"Average TP1 P&L: {tp1_trades['tp1_pnl'].mean()*100:.3f}%")
        
        # Bars held analysis
        print(f"\n⏱️ BARS HELD ANALYSIS:")
        avg_bars_held = trades_df['bars_held'].mean()
        print(f"Average bars held: {avg_bars_held:.1f}")
        for bars in range(1, 6):
            count = len(trades_df[trades_df['bars_held'] == bars])
            print(f"Trades closed in {bars} bar(s): {count}")
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio
        }
    
    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        peak = self.equity_curve[0]
        max_dd = 0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_sharpe_ratio(self, risk_free_rate=0.02):
        """
        Calculate Sharpe ratio using proper formula: (Rp - Rf) / σp
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2% = 0.02)
        
        Returns:
            Annualized Sharpe ratio
        """
        if len(self.equity_curve) < 2:
            return 0
        
        # Calculate periodic returns from equity curve
        returns = []
        for i in range(1, len(self.equity_curve)):
            ret = (self.equity_curve[i] - self.equity_curve[i-1]) / self.equity_curve[i-1]
            returns.append(ret)
        
        if not returns:
            return 0
        
        # Calculate average return and standard deviation
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Convert risk-free rate to periodic rate (assuming 5-minute bars, 24/7 trading)
        # 288 bars per day * 365 days = 105,120 bars per year
        periodic_risk_free_rate = risk_free_rate / (288 * 365)
        
        # Calculate excess return: Rp - Rf
        excess_return = avg_return - periodic_risk_free_rate
        
        # Calculate Sharpe ratio: (Rp - Rf) / σp
        sharpe_ratio = excess_return / std_return
        
        # Annualize the Sharpe ratio
        # Multiply by sqrt(288 * 365) to annualize
        annualized_sharpe = sharpe_ratio * np.sqrt(288 * 365)
        
        return annualized_sharpe

def main():
    # Initialize backtester
    backtester = Edge5RAVFBacktester()
    
    # Run backtest
    results = backtester.run_backtest()
    
    print("\n✅ Backtest complete!")

if __name__ == "__main__":
    main()
