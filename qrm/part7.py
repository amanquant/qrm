"""
Part 7: Expected Shortfall Analysis
=====================================
This script computes Expected Shortfall (ES) for all assets using the
Historical Simulation (non-parametric) approach - the simplest and most
intuitive method.

ES = E[Loss | Loss > VaR] = Average of worst losses beyond VaR

The Historical Simulation approach:
1. Sort historical returns from worst to best
2. Identify the VaR threshold (e.g., 5th percentile for 95% VaR)
3. ES = average of all returns below the VaR threshold
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# 1. DATA LOADING AND PREPARATION
# =============================================================================
def load_and_prepare_data(filepath):
    """Load data from Excel and prepare for analysis."""
    print("=" * 80)
    print("1. DATA LOADING AND PREPARATION")
    print("=" * 80)
    
    asset_names = ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes']
    df = pd.read_excel(filepath, skiprows=2)
    df.columns = ['Date'] + asset_names
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df, asset_names


def compute_log_returns(prices_df):
    """Compute log returns from price data."""
    log_returns = np.log(prices_df / prices_df.shift(1))
    log_returns.dropna(inplace=True)
    log_returns_scaled = log_returns * 100  # Percentage returns
    return log_returns, log_returns_scaled


# =============================================================================
# 2. EXPECTED SHORTFALL - HISTORICAL SIMULATION
# =============================================================================
def compute_historical_es(returns, confidence_levels=[0.95, 0.99]):
    """
    Compute Expected Shortfall using Historical Simulation.
    
    ES (also called CVaR - Conditional VaR) is the expected loss
    given that the loss exceeds VaR.
    
    For a confidence level alpha (e.g., 95%):
    - VaR_alpha = the (1-alpha) quantile of returns (e.g., 5th percentile)
    - ES_alpha = average of returns below VaR_alpha
    
    Parameters:
    -----------
    returns : Series or array
        Return series
    confidence_levels : list
        List of confidence levels (e.g., [0.95, 0.99])
    
    Returns:
    --------
    dict : Dictionary with VaR and ES for each confidence level
    """
    returns_np = np.array(returns)
    n = len(returns_np)
    
    results = {}
    
    for cl in confidence_levels:
        alpha = 1 - cl  # e.g., 0.05 for 95% confidence
        
        # VaR: quantile at alpha level
        var = np.percentile(returns_np, alpha * 100)
        
        # ES: average of returns below VaR
        tail_returns = returns_np[returns_np <= var]
        es = np.mean(tail_returns) if len(tail_returns) > 0 else var
        
        # Number of observations in tail
        n_tail = len(tail_returns)
        
        results[cl] = {
            'VaR': var,
            'ES': es,
            'n_tail': n_tail,
            'ES_VaR_ratio': es / var if var != 0 else 1
        }
    
    return results


def compute_es_all_assets(log_returns_scaled, asset_names, 
                          confidence_levels=[0.95, 0.99]):
    """Compute ES for all assets."""
    print("\n" + "=" * 80)
    print("2. EXPECTED SHORTFALL COMPUTATION (Historical Simulation)")
    print("=" * 80)
    
    print("""
    Method: Historical Simulation (Non-Parametric)
    ===============================================
    This is the simplest approach to compute Expected Shortfall:
    
    1. Take all historical returns
    2. Sort them from worst to best
    3. VaR = the return at the (1-alpha) quantile
    4. ES = average of all returns worse than VaR
    
    Advantages:
    - No distributional assumptions required
    - Easy to understand and implement
    - Captures actual tail behavior
    
    Disadvantages:
    - Requires sufficient historical data
    - Assumes past distribution represents future
    """)
    
    all_results = {}
    
    for asset in asset_names:
        returns = log_returns_scaled[asset]
        es_results = compute_historical_es(returns, confidence_levels)
        all_results[asset] = es_results
    
    return all_results


# =============================================================================
# 3. RESULTS SUMMARY
# =============================================================================
def summarize_es_results(es_results, confidence_levels=[0.95, 0.99]):
    """Create summary tables of ES results."""
    print("\n" + "=" * 80)
    print("3. EXPECTED SHORTFALL SUMMARY")
    print("=" * 80)
    
    for cl in confidence_levels:
        print(f"\n{'-'*60}")
        print(f"Expected Shortfall at {int(cl*100)}% Confidence Level")
        print("-" * 60)
        
        summary_data = []
        for asset, results in es_results.items():
            r = results[cl]
            summary_data.append({
                'Asset': asset,
                'VaR': r['VaR'],
                'ES': r['ES'],
                'ES/VaR Ratio': r['ES_VaR_ratio'],
                'Tail Obs': r['n_tail']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.set_index('Asset', inplace=True)
        print(summary_df.to_string())
        
        print(f"\nInterpretation:")
        print(f"  - VaR: {int((1-cl)*100)}% of days have returns worse than this")
        print(f"  - ES: Average loss on the worst {int((1-cl)*100)}% of days")
        print(f"  - ES/VaR Ratio: How much worse ES is compared to VaR (>1 = fat tails)")
    
    return summary_df


def compare_assets(es_results, confidence_levels=[0.95, 0.99]):
    """Compare ES across assets."""
    print("\n" + "=" * 80)
    print("4. ASSET COMPARISON")
    print("=" * 80)
    
    for cl in confidence_levels:
        print(f"\n{'-'*60}")
        print(f"Ranking at {int(cl*100)}% Confidence Level")
        print("-" * 60)
        
        # Sort by ES (most negative = riskiest)
        es_values = {asset: results[cl]['ES'] for asset, results in es_results.items()}
        sorted_assets = sorted(es_values.items(), key=lambda x: x[1])
        
        print("\nRanking by ES (riskiest to least risky):")
        for rank, (asset, es) in enumerate(sorted_assets, 1):
            var = es_results[asset][cl]['VaR']
            ratio = es_results[asset][cl]['ES_VaR_ratio']
            print(f"  {rank}. {asset}: ES = {es:.4f}%, VaR = {var:.4f}%, Ratio = {ratio:.2f}")


# =============================================================================
# 4. ROLLING EXPECTED SHORTFALL
# =============================================================================
def compute_rolling_es(log_returns_scaled, asset_names, window=252, 
                       confidence_level=0.95):
    """Compute rolling ES over time."""
    print("\n" + "=" * 80)
    print("5. ROLLING EXPECTED SHORTFALL")
    print("=" * 80)
    
    print(f"\nRolling window: {window} days (~1 year)")
    
    rolling_es = {}
    rolling_var = {}
    
    for asset in asset_names:
        returns = log_returns_scaled[asset]
        n = len(returns)
        
        es_series = pd.Series(index=returns.index[window:], dtype=float)
        var_series = pd.Series(index=returns.index[window:], dtype=float)
        
        alpha = 1 - confidence_level
        
        for i in range(window, n):
            window_returns = returns.iloc[i-window:i]
            var = np.percentile(window_returns, alpha * 100)
            tail = window_returns[window_returns <= var]
            es = tail.mean() if len(tail) > 0 else var
            
            es_series.iloc[i - window] = es
            var_series.iloc[i - window] = var
        
        rolling_es[asset] = es_series
        rolling_var[asset] = var_series
    
    rolling_es_df = pd.DataFrame(rolling_es)
    rolling_var_df = pd.DataFrame(rolling_var)
    
    print(f"Rolling ES computed for {len(asset_names)} assets")
    print(f"Date range: {rolling_es_df.index.min()} to {rolling_es_df.index.max()}")
    
    return rolling_es_df, rolling_var_df


# =============================================================================
# 5. VISUALIZATION
# =============================================================================
def plot_es_results(es_results, log_returns_scaled, asset_names,
                    rolling_es_df, rolling_var_df, confidence_level=0.95):
    """Create comprehensive visualizations for ES analysis."""
    print("\n" + "=" * 80)
    print("6. CREATING VISUALIZATIONS")
    print("=" * 80)
    
    cl = confidence_level
    
    # Figure 1: ES and VaR Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, cl_plot in enumerate([0.95, 0.99]):
        ax = axes[idx]
        
        assets = list(es_results.keys())
        var_values = [es_results[a][cl_plot]['VaR'] for a in assets]
        es_values = [es_results[a][cl_plot]['ES'] for a in assets]
        
        x = np.arange(len(assets))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, var_values, width, label='VaR', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, es_values, width, label='ES', color='coral', alpha=0.8)
        
        ax.set_xlabel('Asset')
        ax.set_ylabel('Return (%)')
        ax.set_title(f'VaR vs ES Comparison ({int(cl_plot*100)}% Confidence)')
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45)
        ax.legend()
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_es_var_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nBar chart saved: fig_es_var_comparison.png")
    
    # Figure 2: ES/VaR Ratio Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    assets = list(es_results.keys())
    ratios_95 = [es_results[a][0.95]['ES_VaR_ratio'] for a in assets]
    ratios_99 = [es_results[a][0.99]['ES_VaR_ratio'] for a in assets]
    
    x = np.arange(len(assets))
    width = 0.35
    
    ax.bar(x - width/2, ratios_95, width, label='95% ES/VaR', color='green', alpha=0.7)
    ax.bar(x + width/2, ratios_99, width, label='99% ES/VaR', color='purple', alpha=0.7)
    
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, label='VaR = ES')
    ax.set_xlabel('Asset')
    ax.set_ylabel('ES/VaR Ratio')
    ax.set_title('ES/VaR Ratio Comparison (Higher = Fatter Tails)')
    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_es_var_ratio.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Ratio chart saved: fig_es_var_ratio.png")
    
    # Figure 3: Return Distribution with VaR and ES
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(asset_names):
        ax = axes[idx]
        returns = log_returns_scaled[asset]
        
        # Histogram
        ax.hist(returns, bins=100, density=True, alpha=0.6, color='gray', edgecolor='black')
        
        # VaR and ES lines
        var_95 = es_results[asset][0.95]['VaR']
        es_95 = es_results[asset][0.95]['ES']
        var_99 = es_results[asset][0.99]['VaR']
        es_99 = es_results[asset][0.99]['ES']
        
        ax.axvline(x=var_95, color='blue', linestyle='--', linewidth=2, 
                   label=f'VaR 95%: {var_95:.2f}%')
        ax.axvline(x=es_95, color='red', linestyle='-', linewidth=2, 
                   label=f'ES 95%: {es_95:.2f}%')
        ax.axvline(x=var_99, color='blue', linestyle=':', linewidth=2, 
                   label=f'VaR 99%: {var_99:.2f}%')
        ax.axvline(x=es_99, color='red', linestyle='-.', linewidth=2, 
                   label=f'ES 99%: {es_99:.2f}%')
        
        ax.set_title(f'{asset} Return Distribution')
        ax.set_xlabel('Return (%)')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Shade the tail region
        ax.axvspan(returns.min(), var_95, alpha=0.2, color='red')
    
    plt.tight_layout()
    plt.savefig('fig_es_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Distribution plots saved: fig_es_distributions.png")
    
    # Figure 4: Rolling ES over time
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(asset_names):
        ax = axes[idx]
        
        ax.plot(rolling_var_df.index, rolling_var_df[asset], 
                color='blue', linewidth=0.8, alpha=0.8, label='Rolling VaR')
        ax.plot(rolling_es_df.index, rolling_es_df[asset], 
                color='red', linewidth=0.8, alpha=0.8, label='Rolling ES')
        ax.fill_between(rolling_es_df.index, rolling_var_df[asset], rolling_es_df[asset],
                        alpha=0.3, color='orange', label='ES-VaR Gap')
        
        ax.set_title(f'{asset} - Rolling VaR & ES (95%, 252-day)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_es_rolling.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Rolling ES saved: fig_es_rolling.png")
    
    # Figure 5: ES Comparison Across Assets (Box Plot)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create data for boxplot
    es_data = [log_returns_scaled[asset][log_returns_scaled[asset] <= 
               np.percentile(log_returns_scaled[asset], 5)] 
               for asset in asset_names]
    
    bp = ax.boxplot(es_data, labels=asset_names, patch_artist=True)
    colors = plt.cm.Set3(np.linspace(0, 1, len(asset_names)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_xlabel('Asset')
    ax.set_ylabel('Tail Returns (%)')
    ax.set_title('Distribution of Tail Returns (Worst 5%) by Asset')
    ax.grid(True, alpha=0.3)
    
    # Add ES markers
    for i, asset in enumerate(asset_names):
        es = es_results[asset][0.95]['ES']
        ax.scatter(i + 1, es, color='red', s=100, marker='*', 
                   zorder=5, label='ES' if i == 0 else '')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('fig_es_boxplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Box plot saved: fig_es_boxplot.png")
    
    # Figure 6: Radar Chart Comparison
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    categories = asset_names
    N = len(categories)
    
    # Normalize ES values for radar (make positive for display)
    es_95_norm = [-es_results[a][0.95]['ES'] for a in asset_names]
    es_99_norm = [-es_results[a][0.99]['ES'] for a in asset_names]
    
    # Normalize to 0-1 scale
    max_val = max(max(es_95_norm), max(es_99_norm))
    es_95_norm = [v/max_val for v in es_95_norm]
    es_99_norm = [v/max_val for v in es_99_norm]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    es_95_norm += es_95_norm[:1]
    es_99_norm += es_99_norm[:1]
    
    ax.plot(angles, es_95_norm, 'o-', linewidth=2, label='ES 95%', color='blue')
    ax.fill(angles, es_95_norm, alpha=0.25, color='blue')
    ax.plot(angles, es_99_norm, 'o-', linewidth=2, label='ES 99%', color='red')
    ax.fill(angles, es_99_norm, alpha=0.25, color='red')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Risk Comparison Radar (Higher = More Risk)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('fig_es_radar.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Radar chart saved: fig_es_radar.png")


# =============================================================================
# 6. EXPORT RESULTS
# =============================================================================
def export_results(es_results, rolling_es_df, rolling_var_df, asset_names):
    """Export results to Excel."""
    print("\n" + "=" * 80)
    print("7. EXPORTING RESULTS")
    print("=" * 80)
    
    with pd.ExcelWriter('expected_shortfall_results.xlsx', engine='openpyxl') as writer:
        # Summary
        summary_data = []
        for asset in asset_names:
            for cl in [0.95, 0.99]:
                r = es_results[asset][cl]
                summary_data.append({
                    'Asset': asset,
                    'Confidence': f'{int(cl*100)}%',
                    'VaR': r['VaR'],
                    'ES': r['ES'],
                    'ES_VaR_Ratio': r['ES_VaR_ratio'],
                    'Tail_Observations': r['n_tail']
                })
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
        
        # Rolling ES
        rolling_es_df.to_excel(writer, sheet_name='Rolling_ES')
        rolling_var_df.to_excel(writer, sheet_name='Rolling_VaR')
    
    print("\nResults exported to: expected_shortfall_results.xlsx")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    filepath = 'Manzi.xlsx'
    
    print("\n" + "=" * 80)
    print("EXPECTED SHORTFALL ANALYSIS")
    print("Method: Historical Simulation (Non-Parametric)")
    print("=" * 80 + "\n")
    
    confidence_levels = [0.95, 0.99]
    
    # 1. Load data
    prices_df, asset_names = load_and_prepare_data(filepath)
    
    # 2. Compute log returns
    log_returns, log_returns_scaled = compute_log_returns(prices_df)
    
    # 3. Compute ES for all assets
    es_results = compute_es_all_assets(log_returns_scaled, asset_names, confidence_levels)
    
    # 4. Summarize results
    summary = summarize_es_results(es_results, confidence_levels)
    
    # 5. Compare assets
    compare_assets(es_results, confidence_levels)
    
    # 6. Compute rolling ES
    rolling_es_df, rolling_var_df = compute_rolling_es(
        log_returns_scaled, asset_names, window=252, confidence_level=0.95
    )
    
    # 7. Create visualizations
    plot_es_results(es_results, log_returns_scaled, asset_names,
                    rolling_es_df, rolling_var_df, confidence_level=0.95)
    
    # 8. Export results
    export_results(es_results, rolling_es_df, rolling_var_df, asset_names)
    
    print("\n" + "=" * 80)
    print("EXPECTED SHORTFALL ANALYSIS COMPLETE")
    print("=" * 80)
