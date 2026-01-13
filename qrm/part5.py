"""
Part 5: Additional VaR Methods - GARCH + GPD (Peaks Over Threshold)
====================================================================
This script implements VaR Case 5: GARCH combined with GPD (Generalized Pareto Distribution)
using the Peaks-Over-Threshold (POT) approach for extreme value modeling.

The POT approach:
1. Fit GARCH to capture volatility dynamics
2. Compute standardized residuals
3. Fit GPD to exceedances over a high threshold
4. Combine GARCH variance forecasts with GPD tail quantiles for VaR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
from scipy.stats import genpareto, norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Set display options
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
    
    print(f"Total data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    return df, asset_names


def compute_log_returns(prices_df):
    """Compute log returns from price data."""
    log_returns = np.log(prices_df / prices_df.shift(1))
    log_returns.dropna(inplace=True)
    log_returns_scaled = log_returns * 100
    return log_returns, log_returns_scaled


def split_data(log_returns_scaled, estimation_end_row=5218):
    """Split data into estimation and forecast periods."""
    print("\n" + "=" * 80)
    print("2. DATA SPLITTING")
    print("=" * 80)
    
    estimation_data = log_returns_scaled.iloc[:estimation_end_row]
    forecast_data = log_returns_scaled.iloc[estimation_end_row:]
    
    print(f"Estimation period: {estimation_data.index.min()} to {estimation_data.index.max()}")
    print(f"Estimation sample size: {len(estimation_data)}")
    print(f"Forecast period: {forecast_data.index.min()} to {forecast_data.index.max()}")
    print(f"Forecast sample size: {len(forecast_data)}")
    
    return estimation_data, forecast_data


# =============================================================================
# 2. GARCH ESTIMATION AND STANDARDIZED RESIDUALS
# =============================================================================
def compute_standardized_residuals(estimation_data, asset_names):
    """
    Fit GARCH(1,1) models and extract standardized residuals Z.
    """
    print("\n" + "=" * 80)
    print("3. COMPUTING STANDARDIZED RESIDUALS FROM GARCH")
    print("=" * 80)
    
    garch_models = {}
    std_residuals = {}
    cond_volatilities = {}
    
    for asset in asset_names:
        print(f"\nFitting GARCH(1,1) for {asset}...")
        
        y = estimation_data[asset].dropna()
        
        model = arch_model(y, mean='Constant', vol='GARCH', p=1, q=1, 
                          dist='Normal', rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        garch_models[asset] = result
        std_residuals[asset] = result.std_resid
        cond_volatilities[asset] = result.conditional_volatility
        
        print(f"  Log-likelihood: {result.loglikelihood:.2f}")
        print(f"  Std residuals: mean={result.std_resid.mean():.4f}, std={result.std_resid.std():.4f}")
    
    std_resid_df = pd.DataFrame(std_residuals)
    cond_vol_df = pd.DataFrame(cond_volatilities)
    
    return garch_models, std_resid_df, cond_vol_df


# =============================================================================
# 3. GPD FITTING (PEAKS OVER THRESHOLD)
# =============================================================================
def select_threshold(data, quantile=0.90):
    """
    Select threshold for POT analysis.
    Uses empirical quantile of the data.
    """
    threshold = np.quantile(data, quantile)
    return threshold


def fit_gpd(exceedances):
    """
    Fit GPD to exceedances using MLE.
    
    GPD parameters:
    - xi (shape): determines tail behavior
    - sigma (scale): scaling parameter
    
    scipy.stats.genpareto uses c = xi convention
    """
    if len(exceedances) < 10:
        return {'xi': 0.1, 'sigma': np.std(exceedances), 'n_exceed': len(exceedances)}
    
    # Fit GPD using MLE
    params = genpareto.fit(exceedances, floc=0)
    c, loc, scale = params
    
    return {
        'xi': c,  # shape parameter
        'sigma': scale,  # scale parameter
        'n_exceed': len(exceedances),
        'scipy_params': params
    }


def gpd_var_quantile(gpd_params, threshold, n_total, n_exceed, confidence_level):
    """
    Compute VaR quantile using GPD.
    
    For probability p (e.g., 0.05 for 95% VaR):
    VaR = u + (sigma/xi) * [((n/N_u) * p)^(-xi) - 1]
    
    where u is threshold, n is total obs, N_u is exceedances
    """
    xi = gpd_params['xi']
    sigma = gpd_params['sigma']
    
    p = 1 - confidence_level  # e.g., 0.05 for 95% VaR
    
    # Exceedance probability
    Fu = n_exceed / n_total  # P(X > u)
    
    if abs(xi) < 1e-10:
        # Exponential case (xi = 0)
        q = threshold + sigma * np.log(Fu / p)
    else:
        # General GPD case
        q = threshold + (sigma / xi) * ((Fu / p) ** xi - 1)
    
    return q


def fit_gpd_models(std_resid_df, threshold_quantile=0.90):
    """
    Fit GPD models on standardized residuals for all assets.
    For VaR (left tail), we work with negative residuals.
    """
    print("\n" + "=" * 80)
    print("4. GPD FITTING (PEAKS OVER THRESHOLD)")
    print("=" * 80)
    
    gpd_results = {}
    
    for asset in std_resid_df.columns:
        print(f"\n{'='*60}")
        print(f"Processing {asset}")
        print("=" * 60)
        
        z = std_resid_df[asset].dropna()
        n_total = len(z)
        
        # For left tail VaR, work with -Z
        neg_z = -z
        
        # Select threshold
        threshold = select_threshold(neg_z, threshold_quantile)
        print(f"  Threshold quantile: {threshold_quantile*100:.0f}%")
        print(f"  Threshold value: {threshold:.4f}")
        
        # Extract exceedances
        exceedances = neg_z[neg_z > threshold] - threshold
        n_exceed = len(exceedances)
        
        print(f"  Number of exceedances: {n_exceed} ({n_exceed/n_total*100:.2f}%)")
        
        # Fit GPD
        gpd_params = fit_gpd(exceedances)
        gpd_params['threshold'] = threshold
        gpd_params['n_total'] = n_total
        
        print(f"\n  GPD Parameters:")
        print(f"    xi (shape): {gpd_params['xi']:.4f}")
        print(f"    sigma (scale): {gpd_params['sigma']:.4f}")
        
        # Interpret shape parameter
        if gpd_params['xi'] > 0:
            print(f"    Interpretation: Heavy tail (Frechet-type)")
        elif gpd_params['xi'] < 0:
            print(f"    Interpretation: Bounded tail (Weibull-type)")
        else:
            print(f"    Interpretation: Exponential tail (Gumbel-type)")
        
        gpd_results[asset] = gpd_params
    
    return gpd_results


# =============================================================================
# 4. GARCH-GPD VaR FORECASTING
# =============================================================================
def compute_garch_gpd_var(estimation_data, forecast_data, asset_names,
                          garch_models, gpd_results,
                          confidence_levels=[0.95, 0.99]):
    """
    Compute VaR forecasts combining GARCH variance predictions with GPD quantiles.
    
    VaR_t = mu_t + sigma_t * q_GPD
    
    where:
    - mu_t, sigma_t are one-step-ahead GARCH forecasts
    - q_GPD is the GPD-based quantile for standardized residuals
    """
    print("\n" + "=" * 80)
    print("5. GARCH-GPD VALUE-AT-RISK FORECASTING (Case 5)")
    print("=" * 80)
    
    all_results = {}
    
    for asset in asset_names:
        print(f"\n{'='*60}")
        print(f"Processing {asset}")
        print("=" * 60)
        
        full_data = pd.concat([estimation_data[asset], forecast_data[asset]])
        n_estimation = len(estimation_data)
        n_forecast = len(forecast_data)
        
        # Get GPD parameters
        gpd_params = gpd_results[asset]
        
        # Compute GPD quantiles for standardized residuals
        gpd_quantiles = {}
        for cl in confidence_levels:
            # Quantile for -Z (we need the VaR of -Z, then transform back)
            q_neg_z = gpd_var_quantile(
                gpd_params, 
                gpd_params['threshold'],
                gpd_params['n_total'],
                gpd_params['n_exceed'],
                cl
            )
            # Transform back: quantile of Z = -quantile of -Z
            gpd_quantiles[cl] = -q_neg_z
        
        print(f"  GPD quantiles: 95%={gpd_quantiles[0.95]:.4f}, 99%={gpd_quantiles[0.99]:.4f}")
        
        results = {
            'Date': [],
            'Actual_Return': [],
            'Forecast_Mean': [],
            'Forecast_Variance': [],
            'Forecast_StdDev': [],
        }
        
        for cl in confidence_levels:
            results[f'VaR_GPD_{int(cl*100)}'] = []
        
        print(f"  Computing {n_forecast} one-step-ahead forecasts...")
        
        for i in range(n_forecast):
            forecast_date = forecast_data.index[i]
            actual_return = forecast_data[asset].iloc[i]
            
            # Data up to forecast date
            current_data = full_data.iloc[:n_estimation + i]
            
            # Fit GARCH and forecast
            model = arch_model(current_data, mean='Constant', vol='GARCH',
                              p=1, q=1, dist='Normal', rescale=False)
            res = model.fit(disp='off', show_warning=False, update_freq=0)
            
            forecast = res.forecast(horizon=1)
            mu = forecast.mean.iloc[-1, 0]
            var = forecast.variance.iloc[-1, 0]
            sigma = np.sqrt(var)
            
            results['Date'].append(forecast_date)
            results['Actual_Return'].append(actual_return)
            results['Forecast_Mean'].append(mu)
            results['Forecast_Variance'].append(var)
            results['Forecast_StdDev'].append(sigma)
            
            # Compute VaR using GARCH + GPD
            for cl in confidence_levels:
                q_gpd = gpd_quantiles[cl]
                var_gpd = mu + sigma * q_gpd
                results[f'VaR_GPD_{int(cl*100)}'].append(var_gpd)
        
        results_df = pd.DataFrame(results)
        results_df.set_index('Date', inplace=True)
        
        # Store GPD quantiles
        for cl in confidence_levels:
            results_df.attrs[f'q_gpd_{int(cl*100)}'] = gpd_quantiles[cl]
        
        all_results[asset] = results_df
    
    return all_results


# =============================================================================
# 5. RESULTS SUMMARY AND BACKTESTING
# =============================================================================
def summarize_gpd_var_results(all_results, confidence_levels=[0.95, 0.99]):
    """Summarize GPD-VaR results with backtesting statistics."""
    print("\n" + "=" * 80)
    print("6. GARCH-GPD VaR BACKTESTING RESULTS (Case 5)")
    print("=" * 80)
    
    summary_data = []
    
    for asset, df in all_results.items():
        n_obs = len(df)
        
        for cl in confidence_levels:
            var_col = f'VaR_GPD_{int(cl*100)}'
            
            violations = (df['Actual_Return'] < df[var_col]).sum()
            expected_violations = n_obs * (1 - cl)
            
            # Kupiec's POF test
            p = (1 - cl)
            n = n_obs
            x = violations
            
            if x > 0 and x < n:
                lr_pof = -2 * (np.log((1-p)**(n-x) * p**x) - 
                              np.log((1-x/n)**(n-x) * (x/n)**x))
                pof_pvalue = 1 - stats.chi2.cdf(lr_pof, 1)
            else:
                lr_pof = np.nan
                pof_pvalue = np.nan
            
            summary_data.append({
                'Asset': asset,
                'Confidence': f'{int(cl*100)}%',
                'N_Obs': n_obs,
                'Expected_Viol': expected_violations,
                'Actual_Viol': violations,
                'Violation_Rate': violations / n_obs * 100,
                'Expected_Rate': (1 - cl) * 100,
                'LR_POF': lr_pof,
                'POF_PValue': pof_pvalue
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nVaR Backtesting Summary (GARCH-GPD Method):")
    print("-" * 100)
    print(summary_df.to_string(index=False))
    
    print("\nInterpretation:")
    print("  - VaR Case 5 (GARCH + GPD): Combines GARCH volatility with GPD tail modeling")
    print("  - Violation rate should be close to (1 - Confidence Level)")
    print("  - POF_PValue > 0.05: VaR model is adequate (do not reject)")
    
    return summary_df


def print_gpd_summary(gpd_results):
    """Print summary of GPD parameters."""
    print("\n" + "=" * 80)
    print("GPD PARAMETERS SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for asset, params in gpd_results.items():
        summary_data.append({
            'Asset': asset,
            'Threshold': params['threshold'],
            'N_Exceed': params['n_exceed'],
            'xi (shape)': params['xi'],
            'sigma (scale)': params['sigma']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index('Asset', inplace=True)
    print(summary_df.to_string())
    
    return summary_df


# =============================================================================
# 6. VISUALIZATION
# =============================================================================
def plot_gpd_results(std_resid_df, gpd_results, all_results, asset_names):
    """Create visualizations for GPD analysis."""
    print("\n" + "=" * 80)
    print("7. GPD VISUALIZATIONS")
    print("=" * 80)
    
    # Figure 1: Exceedances and GPD Fit
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(asset_names):
        z = std_resid_df[asset].dropna()
        neg_z = -z
        threshold = gpd_results[asset]['threshold']
        exceedances = neg_z[neg_z > threshold] - threshold
        
        ax = axes[idx]
        if len(exceedances) > 0:
            ax.hist(exceedances, bins=30, density=True, alpha=0.7, 
                    edgecolor='black', label='Exceedances')
            
            # Overlay GPD fit
            x = np.linspace(0.001, exceedances.max(), 100)
            xi = gpd_results[asset]['xi']
            sigma = gpd_results[asset]['sigma']
            pdf = genpareto.pdf(x, xi, scale=sigma)
            ax.plot(x, pdf, 'r-', linewidth=2, label='GPD Fit')
        
        ax.set_title(f'{asset} - Exceedances (GPD Fit)')
        ax.set_xlabel('Exceedance')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_gpd_exceedances.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nExceedances plot saved: fig_gpd_exceedances.png")
    
    # Figure 2: VaR Forecasts
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(asset_names):
        df = all_results[asset]
        ax = axes[idx]
        
        ax.plot(df.index, df['Actual_Return'], 'gray', linewidth=0.5, alpha=0.7, label='Actual')
        ax.plot(df.index, df['VaR_GPD_95'], 'g-', linewidth=1, label='VaR GPD 95%')
        ax.plot(df.index, df['VaR_GPD_99'], 'r--', linewidth=1, label='VaR GPD 99%')
        
        # Mark violations
        violations_95 = df['Actual_Return'] < df['VaR_GPD_95']
        ax.scatter(df.index[violations_95], df['Actual_Return'][violations_95],
                   color='orange', s=20, alpha=0.7, zorder=5)
        
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_title(f'{asset} - GARCH-GPD VaR')
        ax.set_xlabel('Date')
        ax.set_ylabel('Return (%)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_gpd_var_forecasts.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("VaR forecasts saved: fig_gpd_var_forecasts.png")
    
    # Figure 3: Mean Excess Plot (for threshold selection)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(asset_names):
        z = std_resid_df[asset].dropna()
        neg_z = -z.values
        neg_z_sorted = np.sort(neg_z)
        
        thresholds = []
        mean_excesses = []
        
        for i in range(10, len(neg_z_sorted) - 10):
            u = neg_z_sorted[i]
            exceedances = neg_z_sorted[neg_z_sorted > u] - u
            if len(exceedances) > 0:
                thresholds.append(u)
                mean_excesses.append(np.mean(exceedances))
        
        ax = axes[idx]
        ax.plot(thresholds, mean_excesses, 'b-', linewidth=0.8)
        ax.axvline(x=gpd_results[asset]['threshold'], color='r', 
                   linestyle='--', label=f"u={gpd_results[asset]['threshold']:.2f}")
        ax.set_title(f'{asset} - Mean Excess Plot')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Mean Excess')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_gpd_mean_excess.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Mean excess plots saved: fig_gpd_mean_excess.png")


# =============================================================================
# 7. EXPORT RESULTS
# =============================================================================
def export_results(all_results, gpd_summary, summary_df, filename='var_gpd_forecasts.xlsx'):
    """Export results to Excel."""
    print("\n" + "=" * 80)
    print("8. EXPORTING RESULTS")
    print("=" * 80)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for asset, df in all_results.items():
            df.to_excel(writer, sheet_name=asset)
        gpd_summary.to_excel(writer, sheet_name='GPD_Parameters')
        summary_df.to_excel(writer, sheet_name='Backtesting', index=False)
    
    print(f"\nResults exported to: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    filepath = 'Manzi.xlsx'
    
    print("\n" + "=" * 80)
    print("GARCH-GPD VALUE-AT-RISK ANALYSIS")
    print("VaR Case 5: GARCH + GPD (Peaks Over Threshold)")
    print("=" * 80 + "\n")
    
    confidence_levels = [0.95, 0.99]
    threshold_quantile = 0.90  # 90th percentile for POT threshold
    
    # 1. Load data
    prices_df, asset_names = load_and_prepare_data(filepath)
    
    # 2. Compute log returns
    log_returns, log_returns_scaled = compute_log_returns(prices_df)
    
    # 3. Split data
    estimation_data, forecast_data = split_data(log_returns_scaled, 
                                                 estimation_end_row=5218)
    
    # 4. Compute standardized residuals from GARCH
    garch_models, std_resid_df, cond_vol_df = compute_standardized_residuals(
        estimation_data, asset_names
    )
    
    # 5. Fit GPD models on standardized residuals
    gpd_results = fit_gpd_models(std_resid_df, threshold_quantile=threshold_quantile)
    
    # 6. Print GPD summary
    gpd_summary = print_gpd_summary(gpd_results)
    
    # 7. Compute GARCH-GPD VaR forecasts
    all_results = compute_garch_gpd_var(
        estimation_data, forecast_data, asset_names,
        garch_models, gpd_results, confidence_levels
    )
    
    # 8. Summarize and backtest
    summary_df = summarize_gpd_var_results(all_results, confidence_levels)
    
    # 9. Create visualizations
    plot_gpd_results(std_resid_df, gpd_results, all_results, asset_names)
    
    # 10. Export results
    export_results(all_results, gpd_summary, summary_df, 'var_gpd_forecasts.xlsx')
    
    print("\n" + "=" * 80)
    print("GARCH-GPD VaR ANALYSIS COMPLETE")
    print("=" * 80)
