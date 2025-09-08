import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Edge5RAVFAnalyzer:
    def __init__(self, csv_file="edge5_ravf_excursions.csv"):
        self.df = pd.read_csv(csv_file)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['date'] = self.df['timestamp'].dt.date
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.day_name()
        
    def basic_statistics(self):
        """Generate comprehensive basic statistics"""
        print("=" * 80)
        print("COMPREHENSIVE EDGE-5 RAVF STRATEGY ANALYSIS")
        print("=" * 80)
        
        print(f"\n📊 OVERVIEW STATISTICS:")
        print(f"Total Signals: {len(self.df):,}")
        print(f"Date Range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Unique Days: {self.df['date'].nunique()}")
        
        print(f"\n💰 PERFORMANCE METRICS (5 bars after entry):")
        print(f"Average Max Favorable Move (MFE): {self.df['max_favorable'].mean():.3f}%")
        print(f"Average Max Adverse Move (MFA): {self.df['max_adverse'].mean():.3f}%")
        print(f"Standard Deviation MFE: {self.df['max_favorable'].std():.3f}%")
        print(f"Standard Deviation MFA: {self.df['max_adverse'].std():.3f}%")
        print(f"Risk-Reward Ratio: {abs(self.df['max_favorable'].mean() / self.df['max_adverse'].mean()):.3f}")
        
        print(f"\n📈 EXTREME MOVES:")
        print(f"Best Trade: {self.df['max_favorable'].max():.3f}% ({self.df.loc[self.df['max_favorable'].idxmax(), 'signal_type']})")
        print(f"Worst Trade: {self.df['max_adverse'].min():.3f}% ({self.df.loc[self.df['max_adverse'].idxmin(), 'signal_type']})")
        print(f"95th Percentile MFE: {np.percentile(self.df['max_favorable'], 95):.3f}%")
        print(f"95th Percentile MFA: {np.percentile(self.df['max_adverse'], 95):.3f}%")
        
        print(f"\n🔄 RECOVERY ANALYSIS:")
        recovery_trades = self.df[self.df['went_negative_then_profit']]
        print(f"Trades that went negative then recovered: {len(recovery_trades):,} ({len(recovery_trades)/len(self.df)*100:.1f}%)")
        print(f"Average MFA for recovery trades: {recovery_trades['max_adverse'].mean():.3f}%")
        print(f"Average MFE for recovery trades: {recovery_trades['max_favorable'].mean():.3f}%")
        
        return self.df
    
    def create_mfe_mfa_distributions(self):
        """Create Bayesian-style distribution curves for MFE and MFA"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('MFE and MFA Distribution Analysis', fontsize=16, fontweight='bold')
        
        # MFE Distribution
        mfe_data = self.df['max_favorable']
        axes[0,0].hist(mfe_data, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0,0].set_title('MFE Histogram')
        axes[0,0].set_xlabel('Max Favorable Move (%)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].axvline(mfe_data.mean(), color='red', linestyle='--', label=f'Mean: {mfe_data.mean():.3f}%')
        axes[0,0].legend()
        
        # MFA Distribution
        mfa_data = self.df['max_adverse']
        axes[0,1].hist(mfa_data, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0,1].set_title('MFA Histogram')
        axes[0,1].set_xlabel('Max Adverse Move (%)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].axvline(mfa_data.mean(), color='red', linestyle='--', label=f'Mean: {mfa_data.mean():.3f}%')
        axes[0,1].legend()
        
        # MFE KDE Plot
        kde_mfe = gaussian_kde(mfe_data)
        x_range = np.linspace(mfe_data.min(), mfe_data.max(), 200)
        axes[1,0].plot(x_range, kde_mfe(x_range), 'g-', linewidth=2)
        axes[1,0].fill_between(x_range, kde_mfe(x_range), alpha=0.3, color='green')
        axes[1,0].set_title('MFE Kernel Density Estimation')
        axes[1,0].set_xlabel('Max Favorable Move (%)')
        axes[1,0].set_ylabel('Density')
        
        # MFA KDE Plot
        kde_mfa = gaussian_kde(mfa_data)
        x_range = np.linspace(mfa_data.min(), mfa_data.max(), 200)
        axes[1,1].plot(x_range, kde_mfa(x_range), 'r-', linewidth=2)
        axes[1,1].fill_between(x_range, kde_mfa(x_range), alpha=0.3, color='red')
        axes[1,1].set_title('MFA Kernel Density Estimation')
        axes[1,1].set_xlabel('Max Adverse Move (%)')
        axes[1,1].set_ylabel('Density')
        
        plt.tight_layout()
        plt.savefig('mfe_mfa_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_signal_type_analysis(self):
        """Analyze performance by signal type"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis by Signal Type', fontsize=16, fontweight='bold')
        
        # Signal type counts
        signal_counts = self.df['signal_type'].value_counts()
        axes[0,0].bar(signal_counts.index, signal_counts.values, color='skyblue', edgecolor='black')
        axes[0,0].set_title('Signal Type Distribution')
        axes[0,0].set_xlabel('Signal Type')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # MFE by signal type
        mfe_by_type = self.df.groupby('signal_type')['max_favorable'].mean().sort_values(ascending=False)
        axes[0,1].bar(mfe_by_type.index, mfe_by_type.values, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Average MFE by Signal Type')
        axes[0,1].set_xlabel('Signal Type')
        axes[0,1].set_ylabel('Average MFE (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # MFA by signal type
        mfa_by_type = self.df.groupby('signal_type')['max_adverse'].mean().sort_values(ascending=False)
        axes[1,0].bar(mfa_by_type.index, mfa_by_type.values, color='lightcoral', edgecolor='black')
        axes[1,0].set_title('Average MFA by Signal Type')
        axes[1,0].set_xlabel('Signal Type')
        axes[1,0].set_ylabel('Average MFA (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Risk-Reward by signal type
        rr_by_type = (abs(mfe_by_type) / abs(mfa_by_type)).sort_values(ascending=False)
        axes[1,1].bar(rr_by_type.index, rr_by_type.values, color='gold', edgecolor='black')
        axes[1,1].set_title('Risk-Reward Ratio by Signal Type')
        axes[1,1].set_xlabel('Signal Type')
        axes[1,1].set_ylabel('Risk-Reward Ratio')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('signal_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_regime_analysis(self):
        """Analyze performance by market regime"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis by Market Regime', fontsize=16, fontweight='bold')
        
        # Regime distribution
        regime_counts = self.df['regime'].value_counts()
        axes[0,0].pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0,0].set_title('Signal Distribution by Market Regime')
        
        # MFE by regime
        mfe_by_regime = self.df.groupby('regime')['max_favorable'].mean()
        axes[0,1].bar(mfe_by_regime.index, mfe_by_regime.values, color='lightgreen', edgecolor='black')
        axes[0,1].set_title('Average MFE by Market Regime')
        axes[0,1].set_xlabel('Market Regime')
        axes[0,1].set_ylabel('Average MFE (%)')
        
        # MFA by regime
        mfa_by_regime = self.df.groupby('regime')['max_adverse'].mean()
        axes[1,0].bar(mfa_by_regime.index, mfa_by_regime.values, color='lightcoral', edgecolor='black')
        axes[1,0].set_title('Average MFA by Market Regime')
        axes[1,0].set_xlabel('Market Regime')
        axes[1,0].set_ylabel('Average MFA (%)')
        
        # Risk-Reward by regime
        rr_by_regime = (abs(mfe_by_regime) / abs(mfa_by_regime)).sort_values(ascending=False)
        axes[1,1].bar(rr_by_regime.index, rr_by_regime.values, color='gold', edgecolor='black')
        axes[1,1].set_title('Risk-Reward Ratio by Market Regime')
        axes[1,1].set_xlabel('Market Regime')
        axes[1,1].set_ylabel('Risk-Reward Ratio')
        
        plt.tight_layout()
        plt.savefig('regime_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_time_analysis(self):
        """Analyze performance by time of day and day of week"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Analysis by Time', fontsize=16, fontweight='bold')
        
        # MFE by hour
        mfe_by_hour = self.df.groupby('hour')['max_favorable'].mean()
        axes[0,0].plot(mfe_by_hour.index, mfe_by_hour.values, 'g-o', linewidth=2, markersize=6)
        axes[0,0].set_title('Average MFE by Hour of Day')
        axes[0,0].set_xlabel('Hour (UTC)')
        axes[0,0].set_ylabel('Average MFE (%)')
        axes[0,0].grid(True, alpha=0.3)
        
        # MFA by hour
        mfa_by_hour = self.df.groupby('hour')['max_adverse'].mean()
        axes[0,1].plot(mfa_by_hour.index, mfa_by_hour.values, 'r-o', linewidth=2, markersize=6)
        axes[0,1].set_title('Average MFA by Hour of Day')
        axes[0,1].set_xlabel('Hour (UTC)')
        axes[0,1].set_ylabel('Average MFA (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # MFE by day of week
        mfe_by_day = self.df.groupby('day_of_week')['max_favorable'].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        mfe_by_day = mfe_by_day.reindex(day_order)
        axes[1,0].bar(mfe_by_day.index, mfe_by_day.values, color='lightgreen', edgecolor='black')
        axes[1,0].set_title('Average MFE by Day of Week')
        axes[1,0].set_xlabel('Day of Week')
        axes[1,0].set_ylabel('Average MFE (%)')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Signal count by day of week
        count_by_day = self.df.groupby('day_of_week').size()
        count_by_day = count_by_day.reindex(day_order)
        axes[1,1].bar(count_by_day.index, count_by_day.values, color='skyblue', edgecolor='black')
        axes[1,1].set_title('Signal Count by Day of Week')
        axes[1,1].set_xlabel('Day of Week')
        axes[1,1].set_ylabel('Number of Signals')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('time_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_scatter_analysis(self):
        """Create scatter plots showing relationships between variables"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Variable Relationship Analysis', fontsize=16, fontweight='bold')
        
        # MFE vs MFA scatter
        axes[0,0].scatter(self.df['max_adverse'], self.df['max_favorable'], alpha=0.6, s=20)
        axes[0,0].set_xlabel('Max Adverse Move (%)')
        axes[0,0].set_ylabel('Max Favorable Move (%)')
        axes[0,0].set_title('MFE vs MFA Relationship')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(self.df['max_adverse'], self.df['max_favorable'], 1)
        p = np.poly1d(z)
        axes[0,0].plot(self.df['max_adverse'], p(self.df['max_adverse']), "r--", alpha=0.8)
        
        # Entry price vs MFE
        axes[0,1].scatter(self.df['entry_price'], self.df['max_favorable'], alpha=0.6, s=20)
        axes[0,1].set_xlabel('Entry Price')
        axes[0,1].set_ylabel('Max Favorable Move (%)')
        axes[0,1].set_title('Entry Price vs MFE')
        axes[0,1].grid(True, alpha=0.3)
        
        # Entry price vs MFA
        axes[1,0].scatter(self.df['entry_price'], self.df['max_adverse'], alpha=0.6, s=20)
        axes[1,0].set_xlabel('Entry Price')
        axes[1,0].set_ylabel('Max Adverse Move (%)')
        axes[1,0].set_title('Entry Price vs MFA')
        axes[1,0].grid(True, alpha=0.3)
        
        # MFE distribution by regime
        for regime in self.df['regime'].unique():
            regime_data = self.df[self.df['regime'] == regime]
            axes[1,1].hist(regime_data['max_favorable'], alpha=0.6, label=regime, bins=30)
        axes[1,1].set_xlabel('Max Favorable Move (%)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('MFE Distribution by Regime')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('scatter_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_cumulative_analysis(self):
        """Create cumulative performance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Cumulative Performance Analysis', fontsize=16, fontweight='bold')
        
        # Sort by timestamp
        df_sorted = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Cumulative MFE
        cumulative_mfe = df_sorted['max_favorable'].cumsum()
        axes[0,0].plot(range(len(cumulative_mfe)), cumulative_mfe, 'g-', linewidth=2)
        axes[0,0].set_title('Cumulative Max Favorable Moves')
        axes[0,0].set_xlabel('Trade Number')
        axes[0,0].set_ylabel('Cumulative MFE (%)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Cumulative MFA
        cumulative_mfa = df_sorted['max_adverse'].cumsum()
        axes[0,1].plot(range(len(cumulative_mfa)), cumulative_mfa, 'r-', linewidth=2)
        axes[0,1].set_title('Cumulative Max Adverse Moves')
        axes[0,1].set_xlabel('Trade Number')
        axes[0,1].set_ylabel('Cumulative MFA (%)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Net cumulative performance
        net_performance = cumulative_mfe + cumulative_mfa
        axes[1,0].plot(range(len(net_performance)), net_performance, 'b-', linewidth=2)
        axes[1,0].set_title('Net Cumulative Performance (MFE + MFA)')
        axes[1,0].set_xlabel('Trade Number')
        axes[1,0].set_ylabel('Net Performance (%)')
        axes[1,0].grid(True, alpha=0.3)
        axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Rolling average MFE (50-trade window)
        rolling_mfe = df_sorted['max_favorable'].rolling(window=50).mean()
        axes[1,1].plot(range(len(rolling_mfe)), rolling_mfe, 'g-', linewidth=2)
        axes[1,1].set_title('50-Trade Rolling Average MFE')
        axes[1,1].set_xlabel('Trade Number')
        axes[1,1].set_ylabel('Rolling Average MFE (%)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cumulative_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_advanced_statistics(self):
        """Generate advanced statistical analysis"""
        print("\n" + "=" * 80)
        print("ADVANCED STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Normality tests
        print(f"\n📊 NORMALITY TESTS:")
        mfe_normality = stats.normaltest(self.df['max_favorable'])
        mfa_normality = stats.normaltest(self.df['max_adverse'])
        print(f"MFE Normality Test p-value: {mfe_normality.pvalue:.6f}")
        print(f"MFA Normality Test p-value: {mfa_normality.pvalue:.6f}")
        
        # Correlation analysis
        correlation = self.df['max_favorable'].corr(self.df['max_adverse'])
        print(f"\n🔗 CORRELATION ANALYSIS:")
        print(f"MFE vs MFA Correlation: {correlation:.4f}")
        
        # Skewness and kurtosis
        print(f"\n📈 DISTRIBUTION SHAPE:")
        print(f"MFE Skewness: {stats.skew(self.df['max_favorable']):.4f}")
        print(f"MFE Kurtosis: {stats.kurtosis(self.df['max_favorable']):.4f}")
        print(f"MFA Skewness: {stats.skew(self.df['max_adverse']):.4f}")
        print(f"MFA Kurtosis: {stats.kurtosis(self.df['max_adverse']):.4f}")
        
        # Percentile analysis
        print(f"\n📊 PERCENTILE ANALYSIS:")
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("MFE Percentiles:")
        for p in percentiles:
            value = np.percentile(self.df['max_favorable'], p)
            print(f"  {p}th: {value:.3f}%")
        
        print("\nMFA Percentiles:")
        for p in percentiles:
            value = np.percentile(self.df['max_adverse'], p)
            print(f"  {p}th: {value:.3f}%")
        
        # Win rate analysis
        print(f"\n🎯 WIN RATE ANALYSIS:")
        positive_mfe = (self.df['max_favorable'] > 0).sum()
        negative_mfa = (self.df['max_adverse'] < 0).sum()
        print(f"Trades with positive MFE: {positive_mfe:,} ({positive_mfe/len(self.df)*100:.1f}%)")
        print(f"Trades with negative MFA: {negative_mfa:,} ({negative_mfa/len(self.df)*100:.1f}%)")
        
        # Expected value calculation
        expected_mfe = self.df['max_favorable'].mean()
        expected_mfa = self.df['max_adverse'].mean()
        expected_value = expected_mfe + expected_mfa
        print(f"\n💰 EXPECTED VALUE ANALYSIS:")
        print(f"Expected MFE: {expected_mfe:.4f}%")
        print(f"Expected MFA: {expected_mfa:.4f}%")
        print(f"Net Expected Value: {expected_value:.4f}%")
        
        return {
            'correlation': correlation,
            'expected_value': expected_value,
            'mfe_skewness': stats.skew(self.df['max_favorable']),
            'mfa_skewness': stats.skew(self.df['max_adverse'])
        }
    
    def create_interactive_plots(self):
        """Create interactive Plotly plots"""
        # MFE vs MFA interactive scatter
        fig = px.scatter(
            self.df, 
            x='max_adverse', 
            y='max_favorable',
            color='signal_type',
            hover_data=['timestamp', 'entry_price', 'regime'],
            title='Interactive MFE vs MFA Analysis by Signal Type'
        )
        fig.update_layout(width=1000, height=600)
        fig.write_html('interactive_mfe_mfa_scatter.html')
        
        # Time series of MFE and MFA
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['max_favorable'],
            mode='markers',
            name='MFE',
            marker=dict(color='green', size=4)
        ))
        fig.add_trace(go.Scatter(
            x=self.df['timestamp'],
            y=self.df['max_adverse'],
            mode='markers',
            name='MFA',
            marker=dict(color='red', size=4)
        ))
        fig.update_layout(
            title='MFE and MFA Over Time',
            xaxis_title='Time',
            yaxis_title='Move (%)',
            width=1000,
            height=600
        )
        fig.write_html('interactive_time_series.html')
        
        print("Interactive plots saved as HTML files:")
        print("- interactive_mfe_mfa_scatter.html")
        print("- interactive_time_series.html")
    
    def run_complete_analysis(self):
        """Run the complete analysis suite"""
        print("Starting comprehensive EDGE-5 RAVF analysis...")
        
        # Basic statistics
        self.basic_statistics()
        
        # Create all visualizations
        print("\n📊 Creating visualizations...")
        self.create_mfe_mfa_distributions()
        self.create_signal_type_analysis()
        self.create_regime_analysis()
        self.create_time_analysis()
        self.create_scatter_analysis()
        self.create_cumulative_analysis()
        
        # Advanced statistics
        advanced_stats = self.create_advanced_statistics()
        
        # Interactive plots
        print("\n🖱️ Creating interactive plots...")
        self.create_interactive_plots()
        
        print("\n✅ Analysis complete! All charts and interactive plots have been saved.")
        print("📁 Files saved:")
        print("- mfe_mfa_distributions.png")
        print("- signal_type_analysis.png")
        print("- regime_analysis.png")
        print("- time_analysis.png")
        print("- scatter_analysis.png")
        print("- cumulative_analysis.png")
        print("- interactive_mfe_mfa_scatter.html")
        print("- interactive_time_series.html")

def main():
    analyzer = Edge5RAVFAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
