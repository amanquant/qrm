"""
Financial Time Series Analysis
===============================
This script performs comprehensive analysis of asset returns including:
1. Log returns computation
2. Descriptive analysis for:
   - Serial dependence (autocorrelation)
   - Asymmetry in distribution (skewness)
   - Deviations from normality (normality tests)
   - Volatility clustering (ARCH effects)
   - Occurrence of extreme events (kurtosis, VaR)
3. Rolling correlation across asset returns
4. Time-varying stability tests (ADF tests for stationarity of rolling statistics)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, skew, kurtosis
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.6f}'.format)

# Set style for plots
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
    
    # First, read the header row to get asset names
    header_df = pd.read_excel(filepath, nrows=1, header=None)
    
    # Asset names are in the first row (excluding first cell which is 'Name')
    asset_names = ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes']
    
    # Load data, skip first two rows (Code and Currency metadata)
    df = pd.read_excel(filepath, skiprows=2)
    
    # The first column contains dates, rename it
    df.columns = ['Date'] + asset_names
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    # Convert all columns to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop any rows with NaN values
    df.dropna(inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Assets: {df.columns.tolist()}")
    print("\nFirst 5 rows of price data:")
    print(df.head())
    
    return df


# =============================================================================
# 2. COMPUTE LOG RETURNS
# =============================================================================
def compute_log_returns(prices_df):
    """Compute log returns from price data."""
    print("\n" + "=" * 80)
    print("2. LOG RETURNS COMPUTATION")
    print("=" * 80)
    
    # Compute log returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(prices_df / prices_df.shift(1))
    
    # Drop the first row (NaN from shift)
    log_returns.dropna(inplace=True)
    
    print(f"Log returns computed for {len(log_returns)} observations")
    print("\nFirst 5 rows of log returns:")
    print(log_returns.head())
    print("\nBasic statistics of log returns:")
    print(log_returns.describe())
    
    return log_returns


# =============================================================================
# 3. DESCRIPTIVE STATISTICS
# =============================================================================
def descriptive_statistics(log_returns):
    """Compute comprehensive descriptive statistics."""
    print("\n" + "=" * 80)
    print("3. DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    stats_dict = {}
    for col in log_returns.columns:
        data = log_returns[col].dropna()
        stats_dict[col] = {
            'Mean': data.mean(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Skewness': skew(data),
            'Kurtosis': kurtosis(data),  # Excess kurtosis
            'VaR 5%': data.quantile(0.05),
            'VaR 1%': data.quantile(0.01),
            'N': len(data)
        }
    
    stats_df = pd.DataFrame(stats_dict).T
    print("\nDescriptive Statistics Summary:")
    print(stats_df.to_string())
    
    return stats_df


# =============================================================================
# 4. SERIAL DEPENDENCE ANALYSIS
# =============================================================================
def analyze_serial_dependence(log_returns):
    """Analyze serial dependence through autocorrelation tests."""
    print("\n" + "=" * 80)
    print("4. SERIAL DEPENDENCE ANALYSIS")
    print("=" * 80)
    
    print("\n4.1 Autocorrelation Analysis (Ljung-Box Test)")
    print("-" * 60)
    
    results = {}
    for col in log_returns.columns:
        data = log_returns[col].dropna()
        
        # Ljung-Box test for returns
        lb_test_returns = acorr_ljungbox(data, lags=[5, 10, 20], return_df=True)
        
        # Ljung-Box test for squared returns (volatility)
        lb_test_squared = acorr_ljungbox(data**2, lags=[5, 10, 20], return_df=True)
        
        results[col] = {
            'LB_Returns_Lag5': lb_test_returns.loc[5, 'lb_pvalue'],
            'LB_Returns_Lag10': lb_test_returns.loc[10, 'lb_pvalue'],
            'LB_Returns_Lag20': lb_test_returns.loc[20, 'lb_pvalue'],
            'LB_Squared_Lag5': lb_test_squared.loc[5, 'lb_pvalue'],
            'LB_Squared_Lag10': lb_test_squared.loc[10, 'lb_pvalue'],
            'LB_Squared_Lag20': lb_test_squared.loc[20, 'lb_pvalue'],
        }
    
    results_df = pd.DataFrame(results).T
    print("\nLjung-Box Test p-values (H0: No autocorrelation):")
    print(results_df.to_string())
    print("\nInterpretation: p-value < 0.05 indicates significant serial dependence")
    
    return results_df


# =============================================================================
# 5. ASYMMETRY IN DISTRIBUTION (SKEWNESS ANALYSIS)
# =============================================================================
def analyze_asymmetry(log_returns):
    """Analyze asymmetry in the distribution of returns."""
    print("\n" + "=" * 80)
    print("5. ASYMMETRY ANALYSIS (SKEWNESS)")
    print("=" * 80)
    
    results = {}
    for col in log_returns.columns:
        data = log_returns[col].dropna()
        n = len(data)
        skewness = skew(data)
        
        # Standard error of skewness
        se_skew = np.sqrt(6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3)))
        
        # Z-score for skewness
        z_skew = skewness / se_skew
        
        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_skew)))
        
        results[col] = {
            'Skewness': skewness,
            'SE_Skewness': se_skew,
            'Z_Score': z_skew,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        }
    
    results_df = pd.DataFrame(results).T
    print("\nSkewness Test Results (H0: Symmetric distribution):")
    print(results_df.to_string())
    print("\nInterpretation:")
    print("  - Negative skewness: Left tail is longer (more extreme negative returns)")
    print("  - Positive skewness: Right tail is longer (more extreme positive returns)")
    print("  - p-value < 0.05: Distribution is significantly asymmetric")
    
    return results_df


# =============================================================================
# 6. NORMALITY TESTS
# =============================================================================
def test_normality(log_returns):
    """Test for deviations from normality."""
    print("\n" + "=" * 80)
    print("6. NORMALITY TESTS")
    print("=" * 80)
    
    results = {}
    for col in log_returns.columns:
        data = log_returns[col].dropna()
        
        # Jarque-Bera test
        jb_stat, jb_pvalue = jarque_bera(data)
        
        # Shapiro-Wilk test (on subset for large samples)
        sample_size = min(5000, len(data))
        shapiro_sample = data.sample(n=sample_size, random_state=42)
        sw_stat, sw_pvalue = shapiro(shapiro_sample)
        
        # D'Agostino-Pearson test
        dp_stat, dp_pvalue = stats.normaltest(data)
        
        results[col] = {
            'JB_Statistic': jb_stat,
            'JB_PValue': jb_pvalue,
            'Shapiro_Stat': sw_stat,
            'Shapiro_PValue': sw_pvalue,
            'DAgostino_Stat': dp_stat,
            'DAgostino_PValue': dp_pvalue
        }
    
    results_df = pd.DataFrame(results).T
    print("\nNormality Test Results (H0: Data is normally distributed):")
    print(results_df.to_string())
    print("\nInterpretation: p-value < 0.05 indicates significant deviation from normality")
    
    return results_df


# =============================================================================
# 7. VOLATILITY CLUSTERING (ARCH EFFECTS)
# =============================================================================
def analyze_volatility_clustering(log_returns):
    """Analyze volatility clustering through ARCH effects."""
    print("\n" + "=" * 80)
    print("7. VOLATILITY CLUSTERING (ARCH EFFECTS)")
    print("=" * 80)
    
    results = {}
    for col in log_returns.columns:
        data = log_returns[col].dropna()
        
        # Demean returns
        demeaned = data - data.mean()
        
        # ARCH-LM test
        try:
            arch_test = het_arch(demeaned, nlags=5)
            lm_stat = arch_test[0]
            lm_pvalue = arch_test[1]
            f_stat = arch_test[2]
            f_pvalue = arch_test[3]
        except:
            lm_stat = lm_pvalue = f_stat = f_pvalue = np.nan
        
        # Ljung-Box on squared returns
        lb_squared = acorr_ljungbox(data**2, lags=[10], return_df=True)
        lb_pvalue = lb_squared.loc[10, 'lb_pvalue']
        
        results[col] = {
            'ARCH_LM_Stat': lm_stat,
            'ARCH_LM_PValue': lm_pvalue,
            'ARCH_F_Stat': f_stat,
            'ARCH_F_PValue': f_pvalue,
            'LB_Squared_PValue': lb_pvalue
        }
    
    results_df = pd.DataFrame(results).T
    print("\nARCH Effects Test Results (H0: No ARCH effects):")
    print(results_df.to_string())
    print("\nInterpretation: p-value < 0.05 indicates presence of volatility clustering")
    
    return results_df


# =============================================================================
# 8. EXTREME EVENTS ANALYSIS
# =============================================================================
def analyze_extreme_events(log_returns):
    """Analyze the occurrence of extreme events."""
    print("\n" + "=" * 80)
    print("8. EXTREME EVENTS ANALYSIS")
    print("=" * 80)
    
    results = {}
    for col in log_returns.columns:
        data = log_returns[col].dropna()
        
        # Excess kurtosis (normal = 0)
        excess_kurt = kurtosis(data)
        
        # Standard error of kurtosis
        n = len(data)
        se_kurt = np.sqrt(24 * n * (n - 1)**2 / ((n - 3) * (n - 2) * (n + 3) * (n + 5)))
        
        # Z-score for kurtosis
        z_kurt = excess_kurt / se_kurt
        p_kurt = 2 * (1 - stats.norm.cdf(abs(z_kurt)))
        
        # Count extreme events (beyond 3 std devs)
        mean = data.mean()
        std = data.std()
        extreme_pos = (data > mean + 3 * std).sum()
        extreme_neg = (data < mean - 3 * std).sum()
        total_extreme = extreme_pos + extreme_neg
        
        # Expected under normal distribution
        expected_extreme = n * 2 * stats.norm.cdf(-3)  # ~0.27%
        
        # VaR and ES (Expected Shortfall)
        var_95 = data.quantile(0.05)
        var_99 = data.quantile(0.01)
        es_95 = data[data <= var_95].mean()
        es_99 = data[data <= var_99].mean()
        
        results[col] = {
            'Excess_Kurtosis': excess_kurt,
            'Kurt_PValue': p_kurt,
            'Extreme_Positive': extreme_pos,
            'Extreme_Negative': extreme_neg,
            'Total_Extreme': total_extreme,
            'Expected_Extreme': expected_extreme,
            'VaR_95': var_95,
            'VaR_99': var_99,
            'ES_95': es_95,
            'ES_99': es_99
        }
    
    results_df = pd.DataFrame(results).T
    print("\nExtreme Events Analysis (Events beyond 3 std devs):")
    print(results_df.to_string())
    print("\nInterpretation:")
    print("  - Excess Kurtosis > 0: Fat tails (more extreme events than normal)")
    print("  - Total Extreme > Expected: More extreme events than expected under normality")
    
    return results_df


# =============================================================================
# 9. ROLLING CORRELATION ANALYSIS
# =============================================================================
def compute_rolling_correlation(log_returns, window=252):
    """Compute rolling correlation across asset returns."""
    print("\n" + "=" * 80)
    print("9. ROLLING CORRELATION ANALYSIS")
    print("=" * 80)
    
    print(f"\nRolling window: {window} days (~1 year)")
    
    # Compute rolling correlation matrix at each point
    n_assets = len(log_returns.columns)
    assets = log_returns.columns.tolist()
    
    # Create pairs for correlation
    pairs = []
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            pairs.append((assets[i], assets[j]))
    
    # Compute rolling correlation for each pair
    rolling_corr = pd.DataFrame(index=log_returns.index)
    
    for asset1, asset2 in pairs:
        pair_name = f"{asset1}-{asset2}"
        rolling_corr[pair_name] = log_returns[asset1].rolling(window=window).corr(log_returns[asset2])
    
    # Remove NaN values from beginning
    rolling_corr = rolling_corr.dropna()
    
    print(f"\nRolling correlations computed for {len(pairs)} pairs")
    print(f"Date range: {rolling_corr.index.min()} to {rolling_corr.index.max()}")
    
    # Summary statistics of rolling correlations
    print("\nSummary statistics of rolling correlations:")
    summary = rolling_corr.describe()
    print(summary.to_string())
    
    # Current correlation matrix
    current_corr = log_returns.tail(window).corr()
    print(f"\nCurrent correlation matrix (last {window} days):")
    print(current_corr.to_string())
    
    return rolling_corr, current_corr


# =============================================================================
# 10. TIME-VARYING STABILITY TESTS (ADF TESTS)
# =============================================================================
def test_time_varying_stability(log_returns, window=252):
    """
    Test if mean, variance, correlation, kurtosis, and skewness 
    are constant over time using ADF tests on rolling statistics.
    
    ADF test on rolling statistics:
    - H0: Rolling statistic has a unit root (non-stationary, time-varying)
    - H1: Rolling statistic is stationary (constant over time)
    
    If we reject H0 (p < 0.05), the rolling statistic is stationary,
    meaning the underlying parameter tends to revert to a stable level.
    """
    print("\n" + "=" * 80)
    print("10. TIME-VARYING STABILITY TESTS (ADF TEST)")
    print("=" * 80)
    
    print(f"\nRolling window: {window} days")
    print("\nADF Test Interpretation:")
    print("  - H0: Non-stationary (time-varying behavior)")
    print("  - H1: Stationary (mean-reverting, relatively stable)")
    print("  - p-value < 0.05: Reject H0, evidence of stationarity")
    
    results = {}
    
    for col in log_returns.columns:
        data = log_returns[col].dropna()
        asset_results = {}
        
        # Rolling Mean
        rolling_mean = data.rolling(window=window).mean().dropna()
        adf_mean = adfuller(rolling_mean, autolag='AIC')
        asset_results['Mean_ADF_Stat'] = adf_mean[0]
        asset_results['Mean_PValue'] = adf_mean[1]
        
        # Rolling Variance
        rolling_var = data.rolling(window=window).var().dropna()
        adf_var = adfuller(rolling_var, autolag='AIC')
        asset_results['Variance_ADF_Stat'] = adf_var[0]
        asset_results['Variance_PValue'] = adf_var[1]
        
        # Rolling Skewness
        rolling_skew = data.rolling(window=window).skew().dropna()
        adf_skew = adfuller(rolling_skew, autolag='AIC')
        asset_results['Skewness_ADF_Stat'] = adf_skew[0]
        asset_results['Skewness_PValue'] = adf_skew[1]
        
        # Rolling Kurtosis
        rolling_kurt = data.rolling(window=window).kurt().dropna()
        adf_kurt = adfuller(rolling_kurt, autolag='AIC')
        asset_results['Kurtosis_ADF_Stat'] = adf_kurt[0]
        asset_results['Kurtosis_PValue'] = adf_kurt[1]
        
        results[col] = asset_results
    
    results_df = pd.DataFrame(results).T
    print("\n10.1 ADF Test Results for Rolling Statistics (Individual Assets):")
    print(results_df.to_string())
    
    # Rolling correlation ADF tests
    print("\n10.2 ADF Test Results for Rolling Correlations:")
    corr_results = {}
    
    assets = log_returns.columns.tolist()
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            rolling_corr = log_returns[assets[i]].rolling(window=window).corr(log_returns[assets[j]]).dropna()
            adf_corr = adfuller(rolling_corr, autolag='AIC')
            pair_name = f"{assets[i]}-{assets[j]}"
            corr_results[pair_name] = {
                'ADF_Statistic': adf_corr[0],
                'P_Value': adf_corr[1]
            }
    
    corr_results_df = pd.DataFrame(corr_results).T
    print(corr_results_df.to_string())
    
    return results_df, corr_results_df


# =============================================================================
# 11. VISUALIZATION FUNCTIONS
# =============================================================================
def create_visualizations(log_returns, rolling_corr, save_plots=True):
    """Create comprehensive visualizations."""
    print("\n" + "=" * 80)
    print("11. CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Figure 1: Log Returns Time Series
    fig1, axes1 = plt.subplots(3, 2, figsize=(15, 12))
    axes1 = axes1.flatten()
    for idx, col in enumerate(log_returns.columns):
        axes1[idx].plot(log_returns.index, log_returns[col], linewidth=0.5, alpha=0.8)
        axes1[idx].set_title(f'{col} Log Returns')
        axes1[idx].set_xlabel('Date')
        axes1[idx].set_ylabel('Log Return')
        axes1[idx].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if save_plots:
        plt.savefig('fig1_log_returns_timeseries.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 2: Return Distributions
    fig2, axes2 = plt.subplots(3, 2, figsize=(15, 12))
    axes2 = axes2.flatten()
    for idx, col in enumerate(log_returns.columns):
        data = log_returns[col].dropna()
        axes2[idx].hist(data, bins=100, density=True, alpha=0.7, edgecolor='black')
        
        # Overlay normal distribution
        x = np.linspace(data.min(), data.max(), 100)
        normal_pdf = stats.norm.pdf(x, data.mean(), data.std())
        axes2[idx].plot(x, normal_pdf, 'r-', linewidth=2, label='Normal')
        
        axes2[idx].set_title(f'{col} Distribution\nSkew: {skew(data):.3f}, Kurt: {kurtosis(data):.3f}')
        axes2[idx].set_xlabel('Log Return')
        axes2[idx].set_ylabel('Density')
        axes2[idx].legend()
    plt.tight_layout()
    if save_plots:
        plt.savefig('fig2_return_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 3: ACF of Returns and Squared Returns
    fig3, axes3 = plt.subplots(3, 4, figsize=(20, 12))
    for idx, col in enumerate(log_returns.columns):
        data = log_returns[col].dropna()
        
        # ACF of returns
        plot_acf(data, ax=axes3[idx // 2, (idx % 2) * 2], lags=20, title=f'{col} Returns ACF')
        
        # ACF of squared returns
        plot_acf(data**2, ax=axes3[idx // 2, (idx % 2) * 2 + 1], lags=20, title=f'{col} Squared Returns ACF')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('fig3_acf_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 4: Rolling Correlations
    fig4, ax4 = plt.subplots(figsize=(15, 8))
    for col in rolling_corr.columns[:6]:  # Plot first 6 pairs
        ax4.plot(rolling_corr.index, rolling_corr[col], label=col, alpha=0.8)
    ax4.set_title('Rolling Correlations (252-day window)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Correlation')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    if save_plots:
        plt.savefig('fig4_rolling_correlations.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 5: Correlation Heatmap
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    current_corr = log_returns.corr()
    sns.heatmap(current_corr, annot=True, cmap='RdYlBu_r', center=0, 
                ax=ax5, fmt='.3f', square=True)
    ax5.set_title('Current Correlation Matrix (Full Sample)')
    plt.tight_layout()
    if save_plots:
        plt.savefig('fig5_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Figure 6: Rolling Variance (Volatility Clustering Visualization)
    fig6, axes6 = plt.subplots(3, 2, figsize=(15, 12))
    axes6 = axes6.flatten()
    window = 22  # ~1 month
    for idx, col in enumerate(log_returns.columns):
        rolling_vol = log_returns[col].rolling(window=window).std() * np.sqrt(252)
        axes6[idx].plot(rolling_vol.index, rolling_vol, linewidth=0.8, alpha=0.8)
        axes6[idx].set_title(f'{col} Rolling Volatility (22-day, Annualized)')
        axes6[idx].set_xlabel('Date')
        axes6[idx].set_ylabel('Volatility')
    plt.tight_layout()
    if save_plots:
        plt.savefig('fig6_rolling_volatility.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nPlots saved successfully!")


# =============================================================================
# 12. COMPREHENSIVE SUMMARY
# =============================================================================
def generate_summary(desc_stats, serial_results, asymmetry_results, 
                     normality_results, volatility_results, extreme_results,
                     adf_individual, adf_corr):
    """Generate a comprehensive summary of all findings."""
    print("\n" + "=" * 80)
    print("12. COMPREHENSIVE SUMMARY OF FINDINGS")
    print("=" * 80)
    
    print("\n" + "-" * 60)
    print("SERIAL DEPENDENCE:")
    print("-" * 60)
    for asset in serial_results.index:
        lb_returns = serial_results.loc[asset, 'LB_Returns_Lag10']
        lb_squared = serial_results.loc[asset, 'LB_Squared_Lag10']
        returns_sig = "YES" if lb_returns < 0.05 else "NO"
        squared_sig = "YES" if lb_squared < 0.05 else "NO"
        print(f"{asset}: Returns autocorrelation: {returns_sig} (p={lb_returns:.4f}), "
              f"Squared returns: {squared_sig} (p={lb_squared:.4f})")
    
    print("\n" + "-" * 60)
    print("ASYMMETRY (SKEWNESS):")
    print("-" * 60)
    for asset in asymmetry_results.index:
        skewness = asymmetry_results.loc[asset, 'Skewness']
        sig = asymmetry_results.loc[asset, 'Significant']
        direction = "negative (left-skewed)" if skewness < 0 else "positive (right-skewed)"
        print(f"{asset}: Skewness = {skewness:.4f} ({direction}), Significant: {sig}")
    
    print("\n" + "-" * 60)
    print("NORMALITY:")
    print("-" * 60)
    for asset in normality_results.index:
        jb_p = normality_results.loc[asset, 'JB_PValue']
        normal = "NO" if jb_p < 0.05 else "YES"
        print(f"{asset}: Normal distribution: {normal} (Jarque-Bera p={jb_p:.4e})")
    
    print("\n" + "-" * 60)
    print("VOLATILITY CLUSTERING (ARCH EFFECTS):")
    print("-" * 60)
    for asset in volatility_results.index:
        arch_p = volatility_results.loc[asset, 'ARCH_LM_PValue']
        clustering = "YES" if arch_p < 0.05 else "NO"
        print(f"{asset}: Volatility clustering: {clustering} (ARCH-LM p={arch_p:.4e})")
    
    print("\n" + "-" * 60)
    print("EXTREME EVENTS:")
    print("-" * 60)
    for asset in extreme_results.index:
        kurt = extreme_results.loc[asset, 'Excess_Kurtosis']
        total = extreme_results.loc[asset, 'Total_Extreme']
        expected = extreme_results.loc[asset, 'Expected_Extreme']
        fat_tails = "YES" if kurt > 0 else "NO"
        print(f"{asset}: Fat tails: {fat_tails} (Excess Kurt={kurt:.2f}), "
              f"Extreme events: {int(total)} (expected: {expected:.1f})")
    
    print("\n" + "-" * 60)
    print("TIME-VARYING STABILITY (ADF Tests on Rolling Statistics):")
    print("-" * 60)
    print("(Stationary = relatively stable over time, Non-stationary = time-varying)")
    for asset in adf_individual.index:
        mean_p = adf_individual.loc[asset, 'Mean_PValue']
        var_p = adf_individual.loc[asset, 'Variance_PValue']
        skew_p = adf_individual.loc[asset, 'Skewness_PValue']
        kurt_p = adf_individual.loc[asset, 'Kurtosis_PValue']
        
        mean_st = "Stationary" if mean_p < 0.05 else "Non-stationary"
        var_st = "Stationary" if var_p < 0.05 else "Non-stationary"
        skew_st = "Stationary" if skew_p < 0.05 else "Non-stationary"
        kurt_st = "Stationary" if kurt_p < 0.05 else "Non-stationary"
        
        print(f"\n{asset}:")
        print(f"  Mean: {mean_st} (p={mean_p:.4f})")
        print(f"  Variance: {var_st} (p={var_p:.4f})")
        print(f"  Skewness: {skew_st} (p={skew_p:.4f})")
        print(f"  Kurtosis: {kurt_st} (p={kurt_p:.4f})")
    
    print("\n" + "-" * 60)
    print("ROLLING CORRELATION STABILITY:")
    print("-" * 60)
    for pair in adf_corr.index:
        p_val = adf_corr.loc[pair, 'P_Value']
        status = "Stationary" if p_val < 0.05 else "Non-stationary"
        print(f"{pair}: {status} (p={p_val:.4f})")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # File path
    filepath = 'Manzi.xlsx'
    
    print("\n" + "=" * 80)
    print("FINANCIAL TIME SERIES ANALYSIS")
    print("=" * 80 + "\n")
    
    # Load and prepare data
    prices_df = load_and_prepare_data(filepath)
    
    # Compute log returns
    log_returns = compute_log_returns(prices_df)
    
    # Descriptive statistics
    desc_stats = descriptive_statistics(log_returns)
    
    # Serial dependence analysis
    serial_results = analyze_serial_dependence(log_returns)
    
    # Asymmetry analysis
    asymmetry_results = analyze_asymmetry(log_returns)
    
    # Normality tests
    normality_results = test_normality(log_returns)
    
    # Volatility clustering (ARCH effects)
    volatility_results = analyze_volatility_clustering(log_returns)
    
    # Extreme events analysis
    extreme_results = analyze_extreme_events(log_returns)
    
    # Rolling correlation analysis
    rolling_corr, current_corr = compute_rolling_correlation(log_returns)
    
    # Time-varying stability tests (ADF)
    adf_individual, adf_corr = test_time_varying_stability(log_returns)
    
    # Generate comprehensive summary
    generate_summary(desc_stats, serial_results, asymmetry_results,
                    normality_results, volatility_results, extreme_results,
                    adf_individual, adf_corr)
    
    # Create visualizations
    create_visualizations(log_returns, rolling_corr, save_plots=True)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nAll results have been printed and visualizations saved.")
