"""
Part 2: Univariate GARCH Modeling
==================================
This script builds univariate GARCH(1,1) models for each asset using Normal density.
Data is filtered until 31-12-2024 (row 5221 in Excel).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
from scipy import stats
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
def load_and_prepare_data(filepath, max_rows=5219):
    """
    Load data from Excel and prepare for analysis.
    
    Parameters:
    -----------
    filepath : str
        Path to the Excel file
    max_rows : int
        Maximum number of data rows to include (5219 = row 5221 in Excel minus 2 header rows)
        This corresponds to data until 31-12-2024
    """
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
    
    # Filter data until 31-12-2024 (first max_rows observations)
    df = df.iloc[:max_rows]
    
    # Drop any rows with NaN values
    df.dropna(inplace=True)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Assets: {df.columns.tolist()}")
    print("\nLast 5 rows of price data:")
    print(df.tail())
    
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
    
    # Scale returns by 100 for numerical stability in GARCH estimation
    log_returns_scaled = log_returns * 100
    
    print(f"Log returns computed for {len(log_returns)} observations")
    print(f"Date range: {log_returns.index.min()} to {log_returns.index.max()}")
    print("\nBasic statistics of log returns (in %):")
    print(log_returns_scaled.describe())
    
    return log_returns, log_returns_scaled


# =============================================================================
# 3. UNIVARIATE GARCH(1,1) MODEL ESTIMATION
# =============================================================================
def fit_garch_models(log_returns_scaled, p=1, q=1, dist='Normal'):
    """
    Fit GARCH(p,q) models to each asset series.
    
    Parameters:
    -----------
    log_returns_scaled : DataFrame
        Log returns scaled by 100
    p : int
        GARCH lag order (default: 1)
    q : int
        ARCH lag order (default: 1)
    dist : str
        Error distribution ('Normal', 't', 'skewt', 'ged')
    
    Returns:
    --------
    models : dict
        Dictionary of fitted GARCH models
    results_df : DataFrame
        Summary of parameter estimates and significance
    """
    print("\n" + "=" * 80)
    print(f"3. GARCH({p},{q}) MODEL ESTIMATION - {dist} Distribution")
    print("=" * 80)
    
    models = {}
    all_params = []
    
    for asset in log_returns_scaled.columns:
        print(f"\n{'='*60}")
        print(f"Fitting GARCH({p},{q}) for {asset}")
        print("=" * 60)
        
        # Get the series
        y = log_returns_scaled[asset].dropna()
        
        # Specify the model: mean model = constant, volatility model = GARCH(p,q)
        model = arch_model(y, 
                          mean='Constant',  # Constant mean model
                          vol='GARCH',      # GARCH volatility model
                          p=p,              # GARCH lag
                          q=q,              # ARCH lag  
                          dist=dist,        # Normal distribution
                          rescale=False)    # Already scaled
        
        # Fit the model
        result = model.fit(disp='off', show_warning=False)
        models[asset] = result
        
        # Print summary
        print(result.summary())
        
        # Extract parameters and their statistics
        params = result.params
        std_errors = result.std_err
        t_stats = result.tvalues
        p_values = result.pvalues
        
        # Store results
        param_data = {
            'Asset': asset,
            'mu (mean)': params.get('mu', np.nan),
            'mu_pvalue': p_values.get('mu', np.nan),
            'omega': params.get('omega', np.nan),
            'omega_pvalue': p_values.get('omega', np.nan),
            'alpha[1]': params.get('alpha[1]', np.nan),
            'alpha_pvalue': p_values.get('alpha[1]', np.nan),
            'beta[1]': params.get('beta[1]', np.nan),
            'beta_pvalue': p_values.get('beta[1]', np.nan),
            'alpha+beta': params.get('alpha[1]', 0) + params.get('beta[1]', 0),
            'Log-Likelihood': result.loglikelihood,
            'AIC': result.aic,
            'BIC': result.bic
        }
        all_params.append(param_data)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_params)
    results_df.set_index('Asset', inplace=True)
    
    return models, results_df


# =============================================================================
# 4. PARAMETER SIGNIFICANCE ANALYSIS
# =============================================================================
def analyze_parameter_significance(results_df, alpha=0.05):
    """
    Analyze and report on the statistical significance of GARCH parameters.
    
    Parameters:
    -----------
    results_df : DataFrame
        Summary of GARCH parameter estimates
    alpha : float
        Significance level (default: 0.05)
    """
    print("\n" + "=" * 80)
    print("4. PARAMETER SIGNIFICANCE ANALYSIS")
    print("=" * 80)
    
    print(f"\nSignificance level: {alpha}")
    print("Parameters are significant if p-value < {:.2f}".format(alpha))
    
    # Create significance summary
    print("\n" + "-" * 60)
    print("SUMMARY OF PARAMETER ESTIMATES")
    print("-" * 60)
    
    # Display the main results
    display_cols = ['mu (mean)', 'omega', 'alpha[1]', 'beta[1]', 'alpha+beta', 
                   'Log-Likelihood', 'AIC', 'BIC']
    print("\nParameter Estimates:")
    print(results_df[display_cols].to_string())
    
    print("\n" + "-" * 60)
    print("SIGNIFICANCE OF PARAMETERS (p-values)")
    print("-" * 60)
    
    pvalue_cols = ['mu_pvalue', 'omega_pvalue', 'alpha_pvalue', 'beta_pvalue']
    print("\nP-values:")
    print(results_df[pvalue_cols].to_string())
    
    print("\n" + "-" * 60)
    print("INTERPRETATION FOR EACH ASSET")
    print("-" * 60)
    
    for asset in results_df.index:
        print(f"\n{asset}:")
        
        # Mean significance
        mu_p = results_df.loc[asset, 'mu_pvalue']
        mu_sig = "Significant" if mu_p < alpha else "Not significant"
        print(f"  - Mean (mu): {results_df.loc[asset, 'mu (mean)']:.6f} ({mu_sig}, p={mu_p:.4f})")
        
        # Omega (constant in variance equation)
        omega_p = results_df.loc[asset, 'omega_pvalue']
        omega_sig = "Significant" if omega_p < alpha else "Not significant"
        print(f"  - Omega: {results_df.loc[asset, 'omega']:.6f} ({omega_sig}, p={omega_p:.4f})")
        
        # Alpha (ARCH effect)
        alpha_p = results_df.loc[asset, 'alpha_pvalue']
        alpha_sig = "Significant" if alpha_p < alpha else "Not significant"
        print(f"  - Alpha (ARCH): {results_df.loc[asset, 'alpha[1]']:.6f} ({alpha_sig}, p={alpha_p:.4f})")
        
        # Beta (GARCH effect)
        beta_p = results_df.loc[asset, 'beta_pvalue']
        beta_sig = "Significant" if beta_p < alpha else "Not significant"
        print(f"  - Beta (GARCH): {results_df.loc[asset, 'beta[1]']:.6f} ({beta_sig}, p={beta_p:.4f})")
        
        # Persistence
        persistence = results_df.loc[asset, 'alpha+beta']
        stationarity = "Stationary" if persistence < 1 else "Non-stationary (IGARCH)"
        print(f"  - Persistence (alpha+beta): {persistence:.6f} ({stationarity})")
        
        # Model diagnostics
        aic = results_df.loc[asset, 'AIC']
        bic = results_df.loc[asset, 'BIC']
        ll = results_df.loc[asset, 'Log-Likelihood']
        print(f"  - Log-Likelihood: {ll:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}")
    
    # Overall summary
    print("\n" + "-" * 60)
    print("OVERALL FINDINGS")
    print("-" * 60)
    
    # Count significant parameters
    n_assets = len(results_df)
    
    sig_mu = (results_df['mu_pvalue'] < alpha).sum()
    sig_omega = (results_df['omega_pvalue'] < alpha).sum()
    sig_alpha = (results_df['alpha_pvalue'] < alpha).sum()
    sig_beta = (results_df['beta_pvalue'] < alpha).sum()
    
    print(f"\nSignificance summary ({n_assets} assets):")
    print(f"  - Mean (mu) significant in {sig_mu}/{n_assets} assets")
    print(f"  - Omega significant in {sig_omega}/{n_assets} assets")
    print(f"  - Alpha (ARCH) significant in {sig_alpha}/{n_assets} assets")
    print(f"  - Beta (GARCH) significant in {sig_beta}/{n_assets} assets")
    
    # Persistence analysis
    avg_persistence = results_df['alpha+beta'].mean()
    high_persistence = (results_df['alpha+beta'] > 0.9).sum()
    print(f"\nPersistence analysis:")
    print(f"  - Average persistence: {avg_persistence:.4f}")
    print(f"  - Assets with high persistence (>0.9): {high_persistence}/{n_assets}")
    
    return results_df


# =============================================================================
# 5. PLOT CONDITIONAL VARIANCE
# =============================================================================
def plot_conditional_variance(models, log_returns_scaled, assets_to_plot=None):
    """
    Plot the conditional variance (volatility) from GARCH models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of fitted GARCH models
    log_returns_scaled : DataFrame
        Scaled log returns
    assets_to_plot : list or None
        List of assets to plot. If None, plots all assets.
    """
    print("\n" + "=" * 80)
    print("5. CONDITIONAL VARIANCE PLOTS")
    print("=" * 80)
    
    if assets_to_plot is None:
        assets_to_plot = list(models.keys())
    
    n_assets = len(assets_to_plot)
    
    # Figure 1: Conditional Variance for each asset
    if n_assets <= 6:
        fig, axes = plt.subplots(n_assets, 1, figsize=(14, 4 * n_assets))
        if n_assets == 1:
            axes = [axes]
        
        for idx, asset in enumerate(assets_to_plot):
            result = models[asset]
            cond_vol = result.conditional_volatility
            
            # Convert to annualized volatility (from daily, scaled returns)
            # cond_vol is in % terms, annualize by * sqrt(252)
            annualized_vol = cond_vol * np.sqrt(252)
            
            ax = axes[idx]
            ax.plot(cond_vol.index, annualized_vol, color='steelblue', linewidth=0.8, alpha=0.9)
            ax.fill_between(cond_vol.index, 0, annualized_vol, alpha=0.3, color='steelblue')
            ax.set_title(f'{asset} - GARCH(1,1) Conditional Volatility (Annualized)', fontsize=12)
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility (%)')
            ax.grid(True, alpha=0.3)
            
            # Add mean line
            mean_vol = annualized_vol.mean()
            ax.axhline(y=mean_vol, color='red', linestyle='--', alpha=0.7, 
                      label=f'Mean: {mean_vol:.2f}%')
            ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('fig_garch_conditional_volatility_all.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nPlot saved: fig_garch_conditional_volatility_all.png")
    
    # Figure 2: Detailed plot for the first asset with returns overlay
    asset = assets_to_plot[0]
    result = models[asset]
    cond_vol = result.conditional_volatility
    returns = log_returns_scaled[asset]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Subplot 1: Returns
    ax1 = axes[0]
    ax1.plot(returns.index, returns, color='gray', linewidth=0.5, alpha=0.8)
    ax1.set_title(f'{asset} - Log Returns (%)', fontsize=12)
    ax1.set_ylabel('Return (%)')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Conditional Standard Deviation (daily)
    ax2 = axes[1]
    ax2.plot(cond_vol.index, cond_vol, color='steelblue', linewidth=0.8)
    ax2.fill_between(cond_vol.index, 0, cond_vol, alpha=0.3, color='steelblue')
    ax2.set_title(f'{asset} - GARCH(1,1) Conditional Volatility (Daily)', fontsize=12)
    ax2.set_ylabel('Volatility (%)')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Returns with +/- 2 std dev bands
    ax3 = axes[2]
    ax3.plot(returns.index, returns, color='gray', linewidth=0.5, alpha=0.6, label='Returns')
    ax3.plot(cond_vol.index, 2 * cond_vol, color='red', linewidth=0.8, alpha=0.8, label='+2σ')
    ax3.plot(cond_vol.index, -2 * cond_vol, color='red', linewidth=0.8, alpha=0.8, label='-2σ')
    ax3.fill_between(cond_vol.index, -2 * cond_vol, 2 * cond_vol, alpha=0.2, color='red')
    ax3.set_title(f'{asset} - Returns with GARCH ±2σ Bands', fontsize=12)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Return (%)')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fig_garch_detailed_{asset}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nDetailed plot saved: fig_garch_detailed_{asset}.png")
    
    # Figure 3: Comparison of conditional volatilities across assets
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(assets_to_plot)))
    
    for idx, asset in enumerate(assets_to_plot):
        result = models[asset]
        cond_vol = result.conditional_volatility * np.sqrt(252)  # Annualized
        ax.plot(cond_vol.index, cond_vol, color=colors[idx], linewidth=0.8, 
               alpha=0.8, label=asset)
    
    ax.set_title('GARCH(1,1) Conditional Volatility Comparison (Annualized)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility (%)')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_garch_volatility_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nComparison plot saved: fig_garch_volatility_comparison.png")


# =============================================================================
# 6. MODEL DIAGNOSTICS
# =============================================================================
def garch_diagnostics(models, log_returns_scaled):
    """
    Perform diagnostic tests on GARCH standardized residuals.
    """
    print("\n" + "=" * 80)
    print("6. GARCH MODEL DIAGNOSTICS")
    print("=" * 80)
    
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera
    
    diagnostics = []
    
    for asset in models.keys():
        result = models[asset]
        
        # Get standardized residuals
        std_resid = result.std_resid
        
        # Ljung-Box test on standardized residuals
        lb_resid = acorr_ljungbox(std_resid, lags=[10], return_df=True)
        lb_resid_pval = lb_resid.loc[10, 'lb_pvalue']
        
        # Ljung-Box test on squared standardized residuals (check for remaining ARCH)
        lb_sq_resid = acorr_ljungbox(std_resid**2, lags=[10], return_df=True)
        lb_sq_resid_pval = lb_sq_resid.loc[10, 'lb_pvalue']
        
        # Jarque-Bera test on standardized residuals
        jb_stat, jb_pval = jarque_bera(std_resid.dropna())
        
        diagnostics.append({
            'Asset': asset,
            'LB_Residuals_pval': lb_resid_pval,
            'LB_SqResiduals_pval': lb_sq_resid_pval,
            'JB_Statistic': jb_stat,
            'JB_pval': jb_pval,
            'Residuals_OK': 'Yes' if lb_resid_pval > 0.05 else 'No',
            'ARCH_Removed': 'Yes' if lb_sq_resid_pval > 0.05 else 'No',
            'Normality': 'Yes' if jb_pval > 0.05 else 'No'
        })
    
    diag_df = pd.DataFrame(diagnostics)
    diag_df.set_index('Asset', inplace=True)
    
    print("\nDiagnostic Tests on Standardized Residuals:")
    print(diag_df.to_string())
    
    print("\nInterpretation:")
    print("  - Residuals_OK (LB test): No autocorrelation in standardized residuals")
    print("  - ARCH_Removed (LB squared): No remaining ARCH effects")
    print("  - Normality (JB test): Residuals follow normal distribution")
    
    return diag_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    # File path
    filepath = 'Manzi.xlsx'
    
    print("\n" + "=" * 80)
    print("UNIVARIATE GARCH MODELING - NORMAL DISTRIBUTION")
    print("Data until 31-12-2024")
    print("=" * 80 + "\n")
    
    # 1. Load and prepare data (until row 5221 in Excel = 5219 data rows)
    prices_df = load_and_prepare_data(filepath, max_rows=5219)
    
    # 2. Compute log returns
    log_returns, log_returns_scaled = compute_log_returns(prices_df)
    
    # 3. Fit GARCH(1,1) models with Normal distribution
    models, results_df = fit_garch_models(log_returns_scaled, p=1, q=1, dist='Normal')
    
    # 4. Analyze parameter significance
    results_df = analyze_parameter_significance(results_df)
    
    # 5. Plot conditional variance
    plot_conditional_variance(models, log_returns_scaled)
    
    # 6. Model diagnostics
    diag_df = garch_diagnostics(models, log_returns_scaled)
    
    print("\n" + "=" * 80)
    print("GARCH ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nAll results have been printed and visualizations saved.")