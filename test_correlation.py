#!/usr/bin/env python3
"""
Quick test script to run correlation analysis
"""

import subprocess
import sys
import os

def run_backtester():
    """Run backtester with --live flag to generate indicators CSV"""
    print("🔄 Running backtester with --live flag...")
    try:
        result = subprocess.run([sys.executable, "edge5_ravf_backtester.py", "--live"], 
                              capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✅ Backtester completed successfully")
            return True
        else:
            print(f"❌ Backtester failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Backtester timed out")
        return False
    except Exception as e:
        print(f"❌ Error running backtester: {e}")
        return False

def check_files():
    """Check if required files exist"""
    files = {
        'datadoge.csv': 'Live data file',
        'backtest_indicators_with_signals.csv': 'Backtester indicators',
        'live_indicators_log.csv': 'Live trader indicators'
    }
    
    missing_files = []
    for filename, description in files.items():
        if os.path.exists(filename):
            print(f"✅ {description}: {filename}")
        else:
            print(f"❌ {description}: {filename} (missing)")
            missing_files.append(filename)
    
    return len(missing_files) == 0

def main():
    print("🧪 CORRELATION ANALYSIS TEST")
    print("=" * 40)
    
    # Check if datadoge.csv exists
    if not os.path.exists('datadoge.csv'):
        print("❌ datadoge.csv not found!")
        print("   Please ensure the live data file exists before running this test.")
        return
    
    # Run backtester to generate indicators
    if not run_backtester():
        print("❌ Failed to generate backtester indicators")
        return
    
    # Check if live trader has generated indicators
    if not os.path.exists('live_indicators_log.csv'):
        print("⚠️  Live trader indicators not found.")
        print("   You may need to run the live trader to generate live_indicators_log.csv")
        print("   For now, we'll proceed with just the backtester analysis.")
    
    # Run correlation analysis
    print("\n🔄 Running correlation analysis...")
    try:
        result = subprocess.run([sys.executable, "correlation_analysis.py"], 
                              capture_output=True, text=True, timeout=30)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.TimeoutExpired:
        print("⏰ Correlation analysis timed out")
    except Exception as e:
        print(f"❌ Error running correlation analysis: {e}")

if __name__ == "__main__":
    main()
