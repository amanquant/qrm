"""
Part 4: GEV-GARCH Value-at-Risk
================================
This script:
1. Computes standardized residuals Z from GARCH models for in-sample period
2. Fits GEV distribution on standardized residuals:
   - Case 3: GEV with independence assumption
   - Case 4: GEV with dependence (extremal index)
3. Computes VaR forecasts for 2025 combining GARCH predictions with GEV quantiles
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
from scipy.stats import genextreme, norm
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
    Z_t = (r_t - mu) / sigma_t
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
        
        # Fit GARCH(1,1) with Normal distribution
        model = arch_model(y, mean='Constant', vol='GARCH', p=1, q=1, 
                          dist='Normal', rescale=False)
        result = model.fit(disp='off', show_warning=False)
        
        garch_models[asset] = result
        std_residuals[asset] = result.std_resid
        cond_volatilities[asset] = result.conditional_volatility
        
        print(f"  Log-likelihood: {result.loglikelihood:.2f}")
        print(f"  Standardized residuals: mean={result.std_resid.mean():.4f}, "
              f"std={result.std_resid.std():.4f}")
    
    # Create DataFrames
    std_resid_df = pd.DataFrame(std_residuals)
    cond_vol_df = pd.DataFrame(cond_volatilities)
    
    return garch_models, std_resid_df, cond_vol_df


# =============================================================================
# 3. GEV FITTING ON STANDARDIZED RESIDUALS
# =============================================================================
def extract_block_minima(data, block_size=21):
    """
    Extract block minima for GEV fitting.
    For VaR (left tail), we use minima of negative residuals.
    
    Parameters:
    -----------
    data : array-like
        Standardized residuals
    block_size : int
        Block size in days (default: 21 ~ 1 month)
    """
    # For left tail VaR, we work with negative values
    neg_data = -data.values
    
    n_blocks = len(neg_data) // block_size
    block_maxima = []
    
    for i in range(n_blocks):
        block = neg_data[i * block_size:(i + 1) * block_size]
        block_maxima.append(np.max(block))  # Max of -Z = -Min of Z
    
    return np.array(block_maxima)


def estimate_extremal_index(data, threshold_quantile=0.95):
    """
    Estimate the extremal index using the runs estimator.
    
    The extremal index theta in (0,1] measures the degree of clustering
    of extreme events. theta = 1 means independence, theta < 1 means clustering.
    """
    threshold = np.quantile(data, threshold_quantile)
    exceedances = data > threshold
    
    # Count clusters using runs
    n_exceedances = exceedances.sum()
    if n_exceedances == 0:
        return 1.0
    
    # Count number of clusters (runs of exceedances)
    clusters = 0
    in_cluster = False
    
    for exc in exceedances:
        if exc and not in_cluster:
            clusters += 1
            in_cluster = True
        elif not exc:
            in_cluster = False
    
    # Extremal index estimate
    if clusters == 0:
        return 1.0
    
    theta = clusters / n_exceedances
    return min(1.0, max(0.01, theta))


def fit_gev_independence(block_maxima):
    """
    Fit GEV distribution assuming independence (theta = 1).
    """
    # Fit GEV using maximum likelihood
    params = genextreme.fit(block_maxima)
    c, loc, scale = params  # c is -xi (shape), loc is mu, scale is sigma
    
    # Note: scipy uses c = -xi convention
    xi = -c
    mu = loc
    sigma = scale
    
    return {'xi': xi, 'mu': mu, 'sigma': sigma, 'theta': 1.0, 
            'scipy_params': params}


def fit_gev_with_dependence(block_maxima, extremal_index):
    """
    Fit GEV distribution accounting for dependence (theta < 1).
    
    When theta < 1, the effective number of independent observations is reduced,
    affecting the return level calculations.
    """
    # First fit standard GEV
    params = genextreme.fit(block_maxima)
    c, loc, scale = params
    
    xi = -c
    mu = loc
    sigma = scale
    
    return {'xi': xi, 'mu': mu, 'sigma': sigma, 'theta': extremal_index,
            'scipy_params': params}


def fit_gev_models(std_resid_df, block_size=21):
    """
    Fit GEV models on standardized residuals for all assets.
    """
    print("\n" + "=" * 80)
    print("4. GEV FITTING ON STANDARDIZED RESIDUALS")
    print("=" * 80)
    
    gev_results = {}
    
    for asset in std_resid_df.columns:
        print(f"\n{'='*60}")
        print(f"Processing {asset}")
        print("=" * 60)
        
        z = std_resid_df[asset].dropna()
        
        # For left tail VaR, we work with -Z (so minima become maxima)
        neg_z = -z
        
        # Extract block maxima of -Z
        block_maxima = extract_block_minima(z, block_size)
        
        print(f"  Block size: {block_size} days")
        print(f"  Number of blocks: {len(block_maxima)}")
        
        # Estimate extremal index on negative residuals (for left tail)
        theta = estimate_extremal_index(neg_z, threshold_quantile=0.95)
        print(f"  Extremal index (theta): {theta:.4f}")
        
        # Case 3: GEV with independence (theta = 1)
        gev_indep = fit_gev_independence(block_maxima)
        print(f"\n  GEV (Independence, theta=1):")
        print(f"    xi (shape): {gev_indep['xi']:.4f}")
        print(f"    mu (location): {gev_indep['mu']:.4f}")
        print(f"    sigma (scale): {gev_indep['sigma']:.4f}")
        
        # Case 4: GEV with dependence (estimated theta)
        gev_dep = fit_gev_with_dependence(block_maxima, theta)
        print(f"\n  GEV (Dependence, theta={theta:.4f}):")
        print(f"    xi (shape): {gev_dep['xi']:.4f}")
        print(f"    mu (location): {gev_dep['mu']:.4f}")
        print(f"    sigma (scale): {gev_dep['sigma']:.4f}")
        
        gev_results[asset] = {
            'block_maxima': block_maxima,
            'theta': theta,
            'gev_independence': gev_indep,
            'gev_dependence': gev_dep
        }
    
    return gev_results


# =============================================================================
# 4. GEV QUANTILE COMPUTATION
# =============================================================================
def gev_quantile(p, xi, mu, sigma):
    """
    Compute GEV quantile (return level).
    
    For probability p, the quantile is:
    x_p = mu + sigma * ((-log(p))^(-xi) - 1) / xi  if xi != 0
    x_p = mu - sigma * log(-log(p))                if xi = 0
    """
    if abs(xi) < 1e-10:
        return mu - sigma * np.log(-np.log(p))
    else:
        return mu + sigma * ((-np.log(p))**(-xi) - 1) / xi


def compute_gev_var_quantile(gev_params, confidence_level, block_size, 
                              use_extremal_index=False):
    """
    Compute VaR quantile from GEV parameters.
    
    For daily VaR from block maxima:
    - Adjust return period for block size
    - Account for extremal index if using dependence
    """
    xi = gev_params['xi']
    mu = gev_params['mu']
    sigma = gev_params['sigma']
    theta = gev_params['theta'] if use_extremal_index else 1.0
    
    # For VaR, we want the (1-alpha) quantile of the left tail
    # Since we fitted GEV on -Z, we need to transform back
    alpha = 1 - confidence_level  # e.g., 0.05 for 95% VaR
    
    # Return period in blocks
    # Daily exceedance probability = alpha
    # Block exceedance probability ≈ 1 - (1-alpha)^block_size
    # For small alpha: ≈ alpha * block_size
    
    # With extremal index: effective probability is theta * p
    p = 1 - alpha * block_size * theta
    p = max(0.001, min(0.999, p))  # Clip to valid range
    
    # GEV quantile for -Z
    q_neg_z = gev_quantile(p, xi, mu, sigma)
    
    # Transform back: quantile of Z = -quantile of -Z
    q_z = -q_neg_z
    
    return q_z


# =============================================================================
# 5. VaR FORECASTING WITH GARCH-GEV
# =============================================================================
def compute_garch_gev_var(estimation_data, forecast_data, asset_names, 
                          garch_models, gev_results, 
                          confidence_levels=[0.95, 0.99], block_size=21):
    """
    Compute VaR forecasts combining GARCH variance predictions with GEV quantiles.
    
    VaR_t = mu_t + sigma_t * q_GEV
    
    where:
    - mu_t, sigma_t are one-step-ahead GARCH forecasts
    - q_GEV is the GEV quantile for standardized residuals
    """
    print("\n" + "=" * 80)
    print("5. GARCH-GEV VALUE-AT-RISK FORECASTING")
    print("=" * 80)
    
    all_results = {}
    
    for asset in asset_names:
        print(f"\n{'='*60}")
        print(f"Processing {asset}")
        print("=" * 60)
        
        full_data = pd.concat([estimation_data[asset], forecast_data[asset]])
        n_estimation = len(estimation_data)
        n_forecast = len(forecast_data)
        
        # Get GEV quantiles
        gev_indep = gev_results[asset]['gev_independence']
        gev_dep = gev_results[asset]['gev_dependence']
        
        results = {
            'Date': [],
            'Actual_Return': [],
            'Forecast_Mean': [],
            'Forecast_Variance': [],
            'Forecast_StdDev': [],
        }
        
        for cl in confidence_levels:
            # GEV quantiles for standardized residuals
            q_gev_indep = compute_gev_var_quantile(gev_indep, cl, block_size, 
                                                     use_extremal_index=False)
            q_gev_dep = compute_gev_var_quantile(gev_dep, cl, block_size,
                                                   use_extremal_index=True)
            
            results[f'GEV_Quantile_Indep_{int(cl*100)}'] = q_gev_indep
            results[f'GEV_Quantile_Dep_{int(cl*100)}'] = q_gev_dep
            results[f'VaR_GEV_Indep_{int(cl*100)}'] = []
            results[f'VaR_GEV_Dep_{int(cl*100)}'] = []
        
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
            
            # Compute VaR using GARCH + GEV
            for cl in confidence_levels:
                q_indep = results[f'GEV_Quantile_Indep_{int(cl*100)}']
                q_dep = results[f'GEV_Quantile_Dep_{int(cl*100)}']
                
                # VaR Case 3: GARCH + GEV (independence)
                var_gev_indep = mu + sigma * q_indep
                results[f'VaR_GEV_Indep_{int(cl*100)}'].append(var_gev_indep)
                
                # VaR Case 4: GARCH + GEV (dependence/extremal index)
                var_gev_dep = mu + sigma * q_dep
                results[f'VaR_GEV_Dep_{int(cl*100)}'].append(var_gev_dep)
        
        # Create DataFrame
        results_df = pd.DataFrame({k: v for k, v in results.items() 
                                   if not k.startswith('GEV_Quantile')})
        results_df.set_index('Date', inplace=True)
        
        # Store GEV quantiles as attributes
        for cl in confidence_levels:
            results_df.attrs[f'q_gev_indep_{int(cl*100)}'] = results[f'GEV_Quantile_Indep_{int(cl*100)}']
            results_df.attrs[f'q_gev_dep_{int(cl*100)}'] = results[f'GEV_Quantile_Dep_{int(cl*100)}']
        
        all_results[asset] = results_df
        
        print(f"  GEV quantiles (95%): Independence={results['GEV_Quantile_Indep_95']:.4f}, "
              f"Dependence={results['GEV_Quantile_Dep_95']:.4f}")
    
    return all_results


# =============================================================================
# 6. RESULTS SUMMARY
# =============================================================================
def summarize_gev_var_results(all_results, confidence_levels=[0.95, 0.99]):
    """Summarize GEV-VaR results with backtesting statistics."""
    print("\n" + "=" * 80)
    print("6. GEV-VaR BACKTESTING RESULTS")
    print("=" * 80)
    
    summary_data = []
    
    for asset, df in all_results.items():
        n_obs = len(df)
        
        for cl in confidence_levels:
            var_indep_col = f'VaR_GEV_Indep_{int(cl*100)}'
            var_dep_col = f'VaR_GEV_Dep_{int(cl*100)}'
            
            violations_indep = (df['Actual_Return'] < df[var_indep_col]).sum()
            violations_dep = (df['Actual_Return'] < df[var_dep_col]).sum()
            
            expected_violations = n_obs * (1 - cl)
            
            summary_data.append({
                'Asset': asset,
                'Confidence': f'{int(cl*100)}%',
                'N_Obs': n_obs,
                'Expected_Viol': expected_violations,
                'Viol_GEV_Indep': violations_indep,
                'Rate_GEV_Indep': violations_indep / n_obs * 100,
                'Viol_GEV_Dep': violations_dep,
                'Rate_GEV_Dep': violations_dep / n_obs * 100,
                'Expected_Rate': (1 - cl) * 100
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    print("\nVaR Backtesting Summary (GEV Methods):")
    print("-" * 100)
    print(summary_df.to_string(index=False))
    
    print("\nInterpretation:")
    print("  - VaR Case 3 (GEV Independence): Assumes extreme events are independent")
    print("  - VaR Case 4 (GEV Dependence): Accounts for clustering via extremal index")
    print("  - Violation rate should be close to (1 - Confidence Level)")
    
    return summary_df


# =============================================================================
# 7. VISUALIZATION
# =============================================================================
def plot_gev_results(std_resid_df, gev_results, all_results, asset_names):
    """Create visualizations for GEV analysis."""
    print("\n" + "=" * 80)
    print("7. GEV VISUALIZATIONS")
    print("=" * 80)
    
    # Figure 1: Standardized Residuals QQ plots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(asset_names):
        z = std_resid_df[asset].dropna()
        ax = axes[idx]
        stats.probplot(z, dist="norm", plot=ax)
        ax.set_title(f'{asset} - Q-Q Plot (Normal)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_gev_qq_plots.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nQ-Q plots saved: fig_gev_qq_plots.png")
    
    # Figure 2: Block Maxima Histograms with GEV fit
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, asset in enumerate(asset_names):
        block_maxima = gev_results[asset]['block_maxima']
        gev_params = gev_results[asset]['gev_independence']['scipy_params']
        
        ax = axes[idx]
        ax.hist(block_maxima, bins=30, density=True, alpha=0.7, 
                edgecolor='black', label='Block Maxima')
        
        # Overlay GEV fit
        x = np.linspace(block_maxima.min(), block_maxima.max(), 100)
        pdf = genextreme.pdf(x, *gev_params)
        ax.plot(x, pdf, 'r-', linewidth=2, label='GEV Fit')
        
        ax.set_title(f'{asset} - Block Maxima (GEV Fit)')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_gev_block_maxima.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Block maxima plots saved: fig_gev_block_maxima.png")
    
    # Figure 3: VaR Comparison for first asset
    asset = asset_names[0]
    df = all_results[asset]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 95% VaR
    ax1 = axes[0]
    ax1.plot(df.index, df['Actual_Return'], 'gray', linewidth=0.6, alpha=0.7, label='Actual')
    ax1.plot(df.index, df['VaR_GEV_Indep_95'], 'g-', linewidth=1.2, label='VaR GEV Indep (95%)')
    ax1.plot(df.index, df['VaR_GEV_Dep_95'], 'm--', linewidth=1.2, label='VaR GEV Dep (95%)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'{asset} - GEV VaR Comparison (95%)')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 99% VaR
    ax2 = axes[1]
    ax2.plot(df.index, df['Actual_Return'], 'gray', linewidth=0.6, alpha=0.7, label='Actual')
    ax2.plot(df.index, df['VaR_GEV_Indep_99'], 'g-', linewidth=1.2, label='VaR GEV Indep (99%)')
    ax2.plot(df.index, df['VaR_GEV_Dep_99'], 'm--', linewidth=1.2, label='VaR GEV Dep (99%)')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title(f'{asset} - GEV VaR Comparison (99%)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Return (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_gev_var_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("VaR comparison saved: fig_gev_var_comparison.png")


def print_gev_summary(gev_results):
    """Print summary of GEV parameters and extremal indices."""
    print("\n" + "=" * 80)
    print("GEV PARAMETERS SUMMARY")
    print("=" * 80)
    
    summary_data = []
    for asset, res in gev_results.items():
        gev_indep = res['gev_independence']
        gev_dep = res['gev_dependence']
        
        summary_data.append({
            'Asset': asset,
            'Extremal_Index': res['theta'],
            'xi_Indep': gev_indep['xi'],
            'mu_Indep': gev_indep['mu'],
            'sigma_Indep': gev_indep['sigma'],
            'xi_Dep': gev_dep['xi'],
            'mu_Dep': gev_dep['mu'],
            'sigma_Dep': gev_dep['sigma']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.set_index('Asset', inplace=True)
    print(summary_df.to_string())
    
    return summary_df


# =============================================================================
# 8. EXPORT RESULTS
# =============================================================================
def export_results(all_results, gev_summary, filename='var_gev_forecasts.xlsx'):
    """Export results to Excel."""
    print("\n" + "=" * 80)
    print("8. EXPORTING RESULTS")
    print("=" * 80)
    
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for asset, df in all_results.items():
            df.to_excel(writer, sheet_name=asset)
        gev_summary.to_excel(writer, sheet_name='GEV_Parameters')
    
    print(f"\nResults exported to: {filename}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    filepath = 'Manzi.xlsx'
    
    print("\n" + "=" * 80)
    print("GEV-GARCH VALUE-AT-RISK ANALYSIS")
    print("VaR Case 3: GARCH + GEV (Independence)")
    print("VaR Case 4: GARCH + GEV (Dependence/Extremal Index)")
    print("=" * 80 + "\n")
    
    confidence_levels = [0.95, 0.99]
    block_size = 21  # ~1 month
    
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
    
    # 5. Fit GEV models on standardized residuals
    gev_results = fit_gev_models(std_resid_df, block_size=block_size)
    
    # 6. Print GEV summary
    gev_summary = print_gev_summary(gev_results)
    
    # 7. Compute GARCH-GEV VaR forecasts
    all_results = compute_garch_gev_var(
        estimation_data, forecast_data, asset_names,
        garch_models, gev_results, confidence_levels, block_size
    )
    
    # 8. Summarize results
    summary_df = summarize_gev_var_results(all_results, confidence_levels)
    
    # 9. Create visualizations
    plot_gev_results(std_resid_df, gev_results, all_results, asset_names)
    
    # 10. Export results
    export_results(all_results, gev_summary, 'var_gev_forecasts.xlsx')
    
    print("\n" + "=" * 80)
    print("GEV-GARCH VaR ANALYSIS COMPLETE")
    print("=" * 80)
