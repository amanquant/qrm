"""
Part 3: Value-at-Risk Forecasting with GARCH Models
=====================================================
This script uses one-step-ahead (static) forecasts for the last year of the sample
(from 1-1-2025 to 31-12-2025) to compute:
1. VaR Case 1: Value-at-Risk under Normal distribution
2. VaR Case 2: Value-at-Risk under Student's t distribution
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
from scipy.stats import norm, t as student_t
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
    
    # Asset names
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
    
    print(f"Total data shape: {df.shape}")
    print(f"Full date range: {df.index.min()} to {df.index.max()}")
    
    return df, asset_names


def compute_log_returns(prices_df):
    """Compute log returns from price data."""
    # Compute log returns: r_t = ln(P_t / P_{t-1})
    log_returns = np.log(prices_df / prices_df.shift(1))
    log_returns.dropna(inplace=True)
    
    # Scale returns by 100 for numerical stability
    log_returns_scaled = log_returns * 100
    
    return log_returns, log_returns_scaled


def split_data(log_returns_scaled, estimation_end_row=5218, forecast_start_row=5219):
    """
    Split data into estimation and forecast periods.
    
    Estimation period: Until 31-12-2024 (row 5221 in Excel = index 5218 in returns)
    Forecast period: 1-1-2025 to 31-12-2025 (row 5222 to 5482 in Excel)
    """
    print("\n" + "=" * 80)
    print("2. DATA SPLITTING")
    print("=" * 80)
    
    # Estimation sample (for model fitting)
    estimation_data = log_returns_scaled.iloc[:estimation_end_row]
    
    # Forecast sample (for VaR evaluation)
    forecast_data = log_returns_scaled.iloc[forecast_start_row:]
    
    print(f"Estimation period: {estimation_data.index.min()} to {estimation_data.index.max()}")
    print(f"Estimation sample size: {len(estimation_data)}")
    print(f"\nForecast period: {forecast_data.index.min()} to {forecast_data.index.max()}")
    print(f"Forecast sample size: {len(forecast_data)}")
    
    return estimation_data, forecast_data


# =============================================================================
# 2. GARCH MODEL ESTIMATION AND VaR FORECASTING
# =============================================================================
def fit_garch_and_forecast_var(estimation_data, forecast_data, asset, 
                                confidence_levels=[0.95, 0.99]):
    """
    Fit GARCH models and compute one-step-ahead VaR forecasts.
    
    Returns VaR predictions under:
    - Case 1: Normal distribution
    - Case 2: Student's t distribution
    """
    print(f"\n{'='*60}")
    print(f"Processing {asset}")
    print("=" * 60)
    
    # Full data for rolling forecasts
    full_data = pd.concat([estimation_data[asset], forecast_data[asset]])
    
    # Store results
    results = {
        'Date': [],
        'Actual_Return': [],
        'Forecast_Mean': [],
        'Forecast_Variance': [],
        'Forecast_StdDev': [],
    }
    
    # Add VaR columns for each confidence level
    for cl in confidence_levels:
        results[f'VaR_Normal_{int(cl*100)}'] = []
        results[f'VaR_StudentT_{int(cl*100)}'] = []
    
    # Fit models on estimation data first
    # Model 1: GARCH with Normal distribution
    model_normal = arch_model(estimation_data[asset], 
                              mean='Constant', 
                              vol='GARCH', 
                              p=1, q=1, 
                              dist='Normal',
                              rescale=False)
    result_normal = model_normal.fit(disp='off', show_warning=False)
    
    # Model 2: GARCH with Student's t distribution
    model_t = arch_model(estimation_data[asset], 
                         mean='Constant', 
                         vol='GARCH', 
                         p=1, q=1, 
                         dist='t',
                         rescale=False)
    result_t = model_t.fit(disp='off', show_warning=False)
    
    # Get degrees of freedom from t model
    nu = result_t.params.get('nu', 8)  # degrees of freedom
    print(f"  Normal model fitted. Log-likelihood: {result_normal.loglikelihood:.2f}")
    print(f"  Student-t model fitted. Log-likelihood: {result_t.loglikelihood:.2f}, df={nu:.2f}")
    
    # One-step-ahead static forecasts for forecast period
    # Using recursive forecasting: re-estimate model each day with expanding window
    n_estimation = len(estimation_data)
    n_forecast = len(forecast_data)
    
    print(f"  Computing {n_forecast} one-step-ahead forecasts...")
    
    for i in range(n_forecast):
        forecast_date = forecast_data.index[i]
        actual_return = forecast_data[asset].iloc[i]
        
        # Use data up to (but not including) the forecast date
        current_data = full_data.iloc[:n_estimation + i]
        
        # Fit GARCH(1,1) Normal on current data
        model_n = arch_model(current_data, mean='Constant', vol='GARCH', 
                            p=1, q=1, dist='Normal', rescale=False)
        res_n = model_n.fit(disp='off', show_warning=False, update_freq=0)
        
        # Fit GARCH(1,1) Student-t on current data  
        model_t = arch_model(current_data, mean='Constant', vol='GARCH',
                            p=1, q=1, dist='t', rescale=False)
        res_t = model_t.fit(disp='off', show_warning=False, update_freq=0)
        
        # One-step ahead forecast
        forecast_n = res_n.forecast(horizon=1)
        forecast_t = res_t.forecast(horizon=1)
        
        # Get forecast mean and variance
        mu_n = forecast_n.mean.iloc[-1, 0]
        var_n = forecast_n.variance.iloc[-1, 0]
        sigma_n = np.sqrt(var_n)
        
        mu_t = forecast_t.mean.iloc[-1, 0]
        var_t = forecast_t.variance.iloc[-1, 0]
        sigma_t = np.sqrt(var_t)
        
        # Get degrees of freedom from t model
        nu = res_t.params.get('nu', 8)
        
        # Store basic results (using Normal model's forecasts as primary)
        results['Date'].append(forecast_date)
        results['Actual_Return'].append(actual_return)
        results['Forecast_Mean'].append(mu_n)
        results['Forecast_Variance'].append(var_n)
        results['Forecast_StdDev'].append(sigma_n)
        
        # Compute VaR for each confidence level
        for cl in confidence_levels:
            alpha = 1 - cl  # Left tail probability
            
            # VaR Case 1: Normal distribution
            # VaR = mu - z_alpha * sigma
            z_alpha = norm.ppf(alpha)
            var_normal = mu_n + z_alpha * sigma_n
            results[f'VaR_Normal_{int(cl*100)}'].append(var_normal)
            
            # VaR Case 2: Student's t distribution
            # VaR = mu + t_alpha * sigma * sqrt((nu-2)/nu)
            # The scaling factor adjusts for the variance of t distribution
            t_alpha = student_t.ppf(alpha, df=nu)
            # Scale factor: std of t(nu) = sqrt(nu/(nu-2)) for nu > 2
            if nu > 2:
                scale_factor = np.sqrt((nu - 2) / nu)
            else:
                scale_factor = 1
            var_t = mu_t + t_alpha * sigma_t / scale_factor
            results[f'VaR_StudentT_{int(cl*100)}'].append(var_t)
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    results_df.set_index('Date', inplace=True)
    
    return results_df, result_normal, result_t


# =============================================================================
# 3. VaR FORECASTING FOR ALL ASSETS
# =============================================================================
def compute_all_var_forecasts(estimation_data, forecast_data, asset_names,
                              confidence_levels=[0.95, 0.99]):
    """Compute VaR forecasts for all assets."""
    print("\n" + "=" * 80)
    print("3. VALUE-AT-RISK FORECASTING")
    print("=" * 80)
    
    all_results = {}
    all_models_normal = {}
    all_models_t = {}
    
    for asset in asset_names:
        var_df, model_n, model_t = fit_garch_and_forecast_var(
            estimation_data, forecast_data, asset, confidence_levels
        )
        all_results[asset] = var_df
        all_models_normal[asset] = model_n
        all_models_t[asset] = model_t
    
    return all_results, all_models_normal, all_models_t


# =============================================================================
# 4. VaR RESULTS SUMMARY
# =============================================================================
def summarize_var_results(all_results, confidence_levels=[0.95, 0.99]):
    """Summarize VaR results and compute backtesting statistics."""
    print("\n" + "=" * 80)
    print("4. VALUE-AT-RISK RESULTS SUMMARY")
    print("=" * 80)
    
    summary_data = []
    
    for asset, df in all_results.items():
        n_obs = len(df)
        
        for cl in confidence_levels:
            # VaR columns
            var_normal_col = f'VaR_Normal_{int(cl*100)}'
            var_t_col = f'VaR_StudentT_{int(cl*100)}'
            
            # Count violations (actual return < VaR, i.e., losses exceed VaR)
            violations_normal = (df['Actual_Return'] < df[var_normal_col]).sum()
            violations_t = (df['Actual_Return'] < df[var_t_col]).sum()
            
            # Expected violations
            expected_violations = n_obs * (1 - cl)
            
            # Violation rates
            vr_normal = violations_normal / n_obs * 100
            vr_t = violations_t / n_obs * 100
            expected_vr = (1 - cl) * 100
            
            summary_data.append({
                'Asset': asset,
                'Confidence': f'{int(cl*100)}%',
                'N_Obs': n_obs,
                'Expected_Violations': expected_violations,
                'Violations_Normal': violations_normal,
                'ViolationRate_Normal': vr_normal,
                'Violations_StudentT': violations_t,
                'ViolationRate_StudentT': vr_t,
                'Expected_VR': expected_vr
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nVaR Backtesting Summary:")
    print("-" * 100)
    print(summary_df.to_string(index=False))
    
    print("\nInterpretation:")
    print("  - Violation Rate should be close to (1 - Confidence Level)")
    print("  - For 95% VaR: expected violation rate = 5%")
    print("  - For 99% VaR: expected violation rate = 1%")
    print("  - If ViolationRate > Expected: VaR is too conservative (underestimates risk)")
    print("  - If ViolationRate < Expected: VaR may be adequate or too aggressive")
    
    return summary_df


# =============================================================================
# 5. DETAILED VaR STATISTICS
# =============================================================================
def print_var_statistics(all_results, confidence_levels=[0.95, 0.99]):
    """Print detailed VaR statistics for each asset."""
    print("\n" + "=" * 80)
    print("5. DETAILED VaR STATISTICS")
    print("=" * 80)
    
    for cl in confidence_levels:
        print(f"\n{'-'*60}")
        print(f"VaR at {int(cl*100)}% Confidence Level")
        print("-" * 60)
        
        var_normal_col = f'VaR_Normal_{int(cl*100)}'
        var_t_col = f'VaR_StudentT_{int(cl*100)}'
        
        stats_data = []
        for asset, df in all_results.items():
            stats_data.append({
                'Asset': asset,
                'VaR_Normal_Mean': df[var_normal_col].mean(),
                'VaR_Normal_Std': df[var_normal_col].std(),
                'VaR_Normal_Min': df[var_normal_col].min(),
                'VaR_Normal_Max': df[var_normal_col].max(),
                'VaR_StudentT_Mean': df[var_t_col].mean(),
                'VaR_StudentT_Std': df[var_t_col].std(),
                'VaR_StudentT_Min': df[var_t_col].min(),
                'VaR_StudentT_Max': df[var_t_col].max(),
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.set_index('Asset', inplace=True)
        print(stats_df.to_string())


# =============================================================================
# 6. VISUALIZATION
# =============================================================================
def plot_var_results(all_results, assets_to_plot=None, confidence_level=0.95):
    """Plot VaR forecasts vs actual returns."""
    print("\n" + "=" * 80)
    print("6. VaR VISUALIZATIONS")
    print("=" * 80)
    
    if assets_to_plot is None:
        assets_to_plot = list(all_results.keys())
    
    cl = confidence_level
    var_normal_col = f'VaR_Normal_{int(cl*100)}'
    var_t_col = f'VaR_StudentT_{int(cl*100)}'
    
    # Figure 1: VaR plots for each asset
    n_assets = len(assets_to_plot)
    fig, axes = plt.subplots(n_assets, 1, figsize=(14, 4 * n_assets))
    if n_assets == 1:
        axes = [axes]
    
    for idx, asset in enumerate(assets_to_plot):
        df = all_results[asset]
        ax = axes[idx]
        
        # Plot actual returns
        ax.plot(df.index, df['Actual_Return'], 'gray', linewidth=0.6, 
                alpha=0.7, label='Actual Return')
        
        # Plot VaR Normal
        ax.plot(df.index, df[var_normal_col], 'b-', linewidth=1.2, 
                alpha=0.8, label=f'VaR Normal ({int(cl*100)}%)')
        
        # Plot VaR Student-t
        ax.plot(df.index, df[var_t_col], 'r--', linewidth=1.2, 
                alpha=0.8, label=f'VaR Student-t ({int(cl*100)}%)')
        
        # Highlight violations
        violations_normal = df['Actual_Return'] < df[var_normal_col]
        violations_t = df['Actual_Return'] < df[var_t_col]
        
        ax.scatter(df.index[violations_normal], df['Actual_Return'][violations_normal],
                   color='blue', marker='o', s=30, alpha=0.7, label='Violation (Normal)', zorder=5)
        ax.scatter(df.index[violations_t], df['Actual_Return'][violations_t],
                   color='red', marker='x', s=30, alpha=0.7, label='Violation (Student-t)', zorder=5)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'{asset} - VaR Forecasts ({int(cl*100)}% Confidence)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fig_var_forecasts_{int(cl*100)}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nPlot saved: fig_var_forecasts_{int(cl*100)}.png")
    
    # Figure 2: Comparison of VaR Normal vs VaR Student-t for first asset
    asset = assets_to_plot[0]
    df = all_results[asset]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 95% VaR comparison
    ax1 = axes[0]
    ax1.plot(df.index, df['Actual_Return'], 'gray', linewidth=0.6, alpha=0.7, label='Actual Return')
    ax1.plot(df.index, df['VaR_Normal_95'], 'b-', linewidth=1.2, label='VaR Normal (95%)')
    ax1.plot(df.index, df['VaR_StudentT_95'], 'r--', linewidth=1.2, label='VaR Student-t (95%)')
    ax1.fill_between(df.index, df['VaR_Normal_95'], df['VaR_StudentT_95'], 
                     alpha=0.2, color='purple', label='Difference')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'{asset} - 95% VaR Comparison: Normal vs Student-t', fontsize=12)
    ax1.set_ylabel('Return (%)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 99% VaR comparison
    ax2 = axes[1]
    ax2.plot(df.index, df['Actual_Return'], 'gray', linewidth=0.6, alpha=0.7, label='Actual Return')
    ax2.plot(df.index, df['VaR_Normal_99'], 'b-', linewidth=1.2, label='VaR Normal (99%)')
    ax2.plot(df.index, df['VaR_StudentT_99'], 'r--', linewidth=1.2, label='VaR Student-t (99%)')
    ax2.fill_between(df.index, df['VaR_Normal_99'], df['VaR_StudentT_99'], 
                     alpha=0.2, color='purple', label='Difference')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title(f'{asset} - 99% VaR Comparison: Normal vs Student-t', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fig_var_comparison_{asset}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison plot saved: fig_var_comparison_{asset}.png")
    
    # Figure 3: Forecast variance over time
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(assets_to_plot):
        df = all_results[asset]
        ax = axes[idx]
        ax.plot(df.index, df['Forecast_Variance'], 'steelblue', linewidth=0.8)
        ax.fill_between(df.index, 0, df['Forecast_Variance'], alpha=0.3, color='steelblue')
        ax.set_title(f'{asset} - Forecast Variance', fontsize=11)
        ax.set_xlabel('Date')
        ax.set_ylabel('Variance')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_forecast_variance.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Forecast variance plot saved: fig_forecast_variance.png")


# =============================================================================
# 7. EXPORT RESULTS
# =============================================================================
def export_var_results(all_results, filename='var_forecasts.xlsx'):
    """Export VaR forecasts to Excel."""
    print("\n" + "=" * 80)
    print("7. EXPORTING RESULTS")
    print("=" * 80)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for asset, df in all_results.items():
            df.to_excel(writer, sheet_name=asset)
    
    print(f"\nVaR forecasts exported to: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # File path
    filepath = 'Manzi.xlsx'
    
    print("\n" + "=" * 80)
    print("VALUE-AT-RISK FORECASTING WITH GARCH MODELS")
    print("VaR Case 1: Normal Distribution")
    print("VaR Case 2: Student's t Distribution")
    print("Forecast Period: 1-1-2025 to 31-12-2025")
    print("=" * 80 + "\n")
    
    # Confidence levels for VaR
    confidence_levels = [0.95, 0.99]
    
    # 1. Load data
    prices_df, asset_names = load_and_prepare_data(filepath)
    
    # 2. Compute log returns
    log_returns, log_returns_scaled = compute_log_returns(prices_df)
    
    # 3. Split data
    # Row 5221 in Excel = row 5219 in data (after skipping 2 header rows)
    # Returns: row 5218 is last estimation observation (31-12-2024)
    # Row 5219 is first forecast observation (1-1-2025)
    estimation_data, forecast_data = split_data(log_returns_scaled, 
                                                 estimation_end_row=5218,
                                                 forecast_start_row=5218)
    
    # 4. Compute VaR forecasts for all assets
    all_results, models_normal, models_t = compute_all_var_forecasts(
        estimation_data, forecast_data, asset_names, confidence_levels
    )
    
    # 5. Summarize results
    summary_df = summarize_var_results(all_results, confidence_levels)
    
    # 6. Print detailed statistics
    print_var_statistics(all_results, confidence_levels)
    
    # 7. Create visualizations
    plot_var_results(all_results, assets_to_plot=asset_names, confidence_level=0.95)
    
    # 8. Export results
    export_var_results(all_results, 'var_forecasts.xlsx')
    
    print("\n" + "=" * 80)
    print("VaR ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nAll results have been printed and visualizations saved.")
