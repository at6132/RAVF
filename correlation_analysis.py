#!/usr/bin/env python3
"""
Correlation Analysis Script
Compares indicator calculations between backtester and live trader to ensure consistency
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

def load_backtest_indicators():
    """Load indicators from backtester CSV"""
    filename = 'backtest_indicators_with_signals.csv'
    if not os.path.exists(filename):
        print(f"❌ Backtester indicators file not found: {filename}")
        print("   Run: python edge5_ravf_backtester.py --live")
        return None
    
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded backtester indicators: {len(df)} candles")
        return df
    except Exception as e:
        print(f"❌ Error loading backtester indicators: {e}")
        return None

def load_live_indicators():
    """Load indicators from live trader CSV"""
    filename = 'live_indicators_log.csv'
    if not os.path.exists(filename):
        print(f"❌ Live trader indicators file not found: {filename}")
        print("   Run the live trader to generate this file")
        return None
    
    try:
        df = pd.read_csv(filename)
        print(f"✅ Loaded live trader indicators: {len(df)} candles")
        return df
    except Exception as e:
        print(f"❌ Error loading live trader indicators: {e}")
        return None

def align_dataframes(backtest_df, live_df):
    """Align the two dataframes by timestamp for comparison"""
    # Convert timestamp columns to datetime
    backtest_df['timestamp'] = pd.to_datetime(backtest_df['timestamp'])
    live_df['timestamp'] = pd.to_datetime(live_df['timestamp'])
    
    # Merge on timestamp to align data
    merged = pd.merge(backtest_df, live_df, on='timestamp', suffixes=('_backtest', '_live'))
    
    print(f"📊 Aligned data: {len(merged)} matching timestamps")
    return merged

def calculate_correlations(merged_df):
    """Calculate correlations between backtester and live trader indicators"""
    # Define indicator columns to compare
    indicators = ['atr', 'vwap', 'clv', 'zvol', 'entropy']
    
    correlations = {}
    differences = {}
    
    for indicator in indicators:
        backtest_col = f"{indicator}_backtest"
        live_col = f"{indicator}_live"
        
        if backtest_col in merged_df.columns and live_col in merged_df.columns:
            # Remove NaN values for correlation calculation
            valid_data = merged_df[[backtest_col, live_col]].dropna()
            
            if len(valid_data) > 0:
                # Calculate correlation
                corr = valid_data[backtest_col].corr(valid_data[live_col])
                correlations[indicator] = corr
                
                # Calculate mean absolute difference
                diff = np.mean(np.abs(valid_data[backtest_col] - valid_data[live_col]))
                differences[indicator] = diff
                
                print(f"📈 {indicator.upper()}:")
                print(f"   Correlation: {corr:.6f}")
                print(f"   Mean Abs Diff: {diff:.8f}")
                print(f"   Valid samples: {len(valid_data)}")
            else:
                print(f"⚠️  {indicator.upper()}: No valid data for correlation")
        else:
            print(f"❌ {indicator.upper()}: Column not found in merged data")
    
    return correlations, differences

def create_correlation_matrix(merged_df):
    """Create a correlation matrix visualization"""
    # Define indicator columns to compare
    indicators = ['atr', 'vwap', 'clv', 'zvol', 'entropy']
    
    # Create correlation matrix
    corr_data = []
    for indicator in indicators:
        backtest_col = f"{indicator}_backtest"
        live_col = f"{indicator}_live"
        
        if backtest_col in merged_df.columns and live_col in merged_df.columns:
            valid_data = merged_df[[backtest_col, live_col]].dropna()
            if len(valid_data) > 0:
                corr = valid_data[backtest_col].corr(valid_data[live_col])
                corr_data.append([indicator, corr])
    
    if corr_data:
        corr_df = pd.DataFrame(corr_data, columns=['Indicator', 'Correlation'])
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Correlation bar chart
        plt.subplot(2, 2, 1)
        bars = plt.bar(corr_df['Indicator'], corr_df['Correlation'], 
                      color=['green' if x > 0.99 else 'orange' if x > 0.95 else 'red' for x in corr_df['Correlation']])
        plt.title('Indicator Correlation: Backtester vs Live Trader')
        plt.ylabel('Correlation Coefficient')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, corr_df['Correlation']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.4f}', ha='center', va='bottom')
        
        # Scatter plots for each indicator
        for i, indicator in enumerate(indicators):
            if i < 3:  # Only show first 3 scatter plots (subplot 2, 3, 4)
                plt.subplot(2, 2, i + 2)
                backtest_col = f"{indicator}_backtest"
                live_col = f"{indicator}_live"
                
                if backtest_col in merged_df.columns and live_col in merged_df.columns:
                    valid_data = merged_df[[backtest_col, live_col]].dropna()
                    if len(valid_data) > 0:
                        plt.scatter(valid_data[backtest_col], valid_data[live_col], alpha=0.6)
                        plt.xlabel(f'Backtester {indicator.upper()}')
                        plt.ylabel(f'Live Trader {indicator.upper()}')
                        plt.title(f'{indicator.upper()} Comparison')
                        
                        # Add diagonal line for perfect correlation
                        min_val = min(valid_data[backtest_col].min(), valid_data[live_col].min())
                        max_val = max(valid_data[backtest_col].max(), valid_data[live_col].max())
                        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Correlation analysis saved as 'correlation_analysis.png'")
        
        return corr_df
    else:
        print("❌ No correlation data available")
        return None

def analyze_differences(merged_df):
    """Analyze specific differences between backtester and live trader"""
    print("\n🔍 DETAILED DIFFERENCE ANALYSIS")
    print("=" * 50)
    
    indicators = ['atr', 'vwap', 'clv', 'zvol', 'entropy']
    
    for indicator in indicators:
        backtest_col = f"{indicator}_backtest"
        live_col = f"{indicator}_live"
        
        if backtest_col in merged_df.columns and live_col in merged_df.columns:
            valid_data = merged_df[[backtest_col, live_col]].dropna()
            
            if len(valid_data) > 0:
                print(f"\n📊 {indicator.upper()} Analysis:")
                print(f"   Backtester mean: {valid_data[backtest_col].mean():.8f}")
                print(f"   Live trader mean: {valid_data[live_col].mean():.8f}")
                print(f"   Mean difference: {np.mean(valid_data[backtest_col] - valid_data[live_col]):.8f}")
                print(f"   Max difference: {np.max(np.abs(valid_data[backtest_col] - valid_data[live_col])):.8f}")
                print(f"   Std difference: {np.std(valid_data[backtest_col] - valid_data[live_col]):.8f}")
                
                # Check for perfect matches
                perfect_matches = np.sum(valid_data[backtest_col] == valid_data[live_col])
                print(f"   Perfect matches: {perfect_matches}/{len(valid_data)} ({perfect_matches/len(valid_data)*100:.1f}%)")
                
                # Check for very close matches (within 1e-10)
                close_matches = np.sum(np.abs(valid_data[backtest_col] - valid_data[live_col]) < 1e-10)
                print(f"   Close matches (1e-10): {close_matches}/{len(valid_data)} ({close_matches/len(valid_data)*100:.1f}%)")

def main():
    """Main analysis function"""
    print("🔍 CORRELATION ANALYSIS: Backtester vs Live Trader")
    print("=" * 60)
    
    # Load data
    backtest_df = load_backtest_indicators()
    live_df = load_live_indicators()
    
    if backtest_df is None or live_df is None:
        print("❌ Cannot proceed without both CSV files")
        return
    
    # Align dataframes
    merged_df = align_dataframes(backtest_df, live_df)
    
    if len(merged_df) == 0:
        print("❌ No matching timestamps found between backtester and live trader")
        return
    
    # Calculate correlations
    print("\n📈 CORRELATION ANALYSIS")
    print("=" * 30)
    correlations, differences = calculate_correlations(merged_df)
    
    # Create visualization
    print("\n📊 Creating correlation matrix...")
    corr_df = create_correlation_matrix(merged_df)
    
    # Analyze differences
    analyze_differences(merged_df)
    
    # Summary
    print("\n📋 SUMMARY")
    print("=" * 20)
    if correlations:
        avg_corr = np.mean(list(correlations.values()))
        print(f"Average correlation: {avg_corr:.6f}")
        
        if avg_corr > 0.99:
            print("✅ EXCELLENT: Indicators are nearly identical")
        elif avg_corr > 0.95:
            print("✅ GOOD: Indicators are very similar")
        elif avg_corr > 0.90:
            print("⚠️  FAIR: Indicators are reasonably similar")
        else:
            print("❌ POOR: Indicators differ significantly")
    
    print(f"\n📁 Files analyzed:")
    print(f"   Backtester: {len(backtest_df)} candles")
    print(f"   Live trader: {len(live_df)} candles")
    print(f"   Aligned: {len(merged_df)} candles")

if __name__ == "__main__":
    main()