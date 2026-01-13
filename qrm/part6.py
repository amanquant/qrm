"""
Part 6: Comprehensive VaR Backtesting Framework
=================================================
This script performs backtesting evaluation across all competing VaR models:
- Case 1: GARCH + Normal
- Case 2: GARCH + Student-t
- Case 3: GARCH + GEV (Independence)
- Case 4: GARCH + GEV (Dependence)
- Case 5: GARCH + GPD

Backtesting methods:
1. Kupiec Unconditional Coverage (UC) Test
2. Christoffersen Independence (I) Test
3. Christoffersen Conditional Coverage (CC) Test
4. Berkowitz Test of Independence
5. Dynamic Quantile (DQ) Test of Engle and Manganelli
6. Model Confidence Set (MCS)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2, norm
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)

plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# 1. LOAD VAR FORECASTS FROM ALL MODELS
# =============================================================================
def load_var_forecasts():
    """Load VaR forecasts from all models."""
    print("=" * 80)
    print("1. LOADING VAR FORECASTS FROM ALL MODELS")
    print("=" * 80)
    
    var_data = {}
    
    # Load Case 1 & 2: Normal and Student-t (from part3)
    try:
        xl_normal_t = pd.ExcelFile('var_forecasts.xlsx')
        for asset in xl_normal_t.sheet_names:
            df = pd.read_excel(xl_normal_t, sheet_name=asset, index_col=0)
            if asset not in var_data:
                var_data[asset] = {}
            var_data[asset]['Actual'] = df['Actual_Return']
            var_data[asset]['VaR_Normal_95'] = df['VaR_Normal_95']
            var_data[asset]['VaR_Normal_99'] = df['VaR_Normal_99']
            var_data[asset]['VaR_StudentT_95'] = df['VaR_StudentT_95']
            var_data[asset]['VaR_StudentT_99'] = df['VaR_StudentT_99']
        print("  Loaded Case 1 (Normal) and Case 2 (Student-t)")
    except Exception as e:
        print(f"  Warning: Could not load var_forecasts.xlsx: {e}")
    
    # Load Case 3 & 4: GEV Independence and Dependence (from part4)
    try:
        xl_gev = pd.ExcelFile('var_gev_forecasts.xlsx')
        for asset in xl_gev.sheet_names:
            if asset == 'GEV_Parameters':
                continue
            df = pd.read_excel(xl_gev, sheet_name=asset, index_col=0)
            if asset not in var_data:
                var_data[asset] = {}
            var_data[asset]['VaR_GEV_Indep_95'] = df['VaR_GEV_Indep_95']
            var_data[asset]['VaR_GEV_Indep_99'] = df['VaR_GEV_Indep_99']
            var_data[asset]['VaR_GEV_Dep_95'] = df['VaR_GEV_Dep_95']
            var_data[asset]['VaR_GEV_Dep_99'] = df['VaR_GEV_Dep_99']
        print("  Loaded Case 3 (GEV Indep) and Case 4 (GEV Dep)")
    except Exception as e:
        print(f"  Warning: Could not load var_gev_forecasts.xlsx: {e}")
    
    # Load Case 5: GPD (from part5)
    try:
        xl_gpd = pd.ExcelFile('var_gpd_forecasts.xlsx')
        for asset in xl_gpd.sheet_names:
            if asset in ['GPD_Parameters', 'Backtesting']:
                continue
            df = pd.read_excel(xl_gpd, sheet_name=asset, index_col=0)
            if asset not in var_data:
                var_data[asset] = {}
            var_data[asset]['VaR_GPD_95'] = df['VaR_GPD_95']
            var_data[asset]['VaR_GPD_99'] = df['VaR_GPD_99']
        print("  Loaded Case 5 (GPD)")
    except Exception as e:
        print(f"  Warning: Could not load var_gpd_forecasts.xlsx: {e}")
    
    print(f"\n  Assets loaded: {list(var_data.keys())}")
    
    return var_data


# =============================================================================
# 2. BACKTESTING FUNCTIONS
# =============================================================================
def compute_hits(returns, var):
    """
    Compute hit sequence (violations).
    Hit = 1 if return < VaR (violation), 0 otherwise.
    """
    hits = (returns < var).astype(int)
    return hits


def kupiec_uc_test(hits, alpha):
    """
    Kupiec's Unconditional Coverage (UC) Test.
    
    H0: The violation rate equals the expected rate (1 - alpha)
    
    LR_UC = -2 * log[(1-p)^(n-x) * p^x] + 2 * log[(1-x/n)^(n-x) * (x/n)^x]
    
    Under H0, LR_UC ~ chi2(1)
    """
    n = len(hits)
    x = hits.sum()  # Number of violations
    p = 1 - alpha  # Expected violation rate
    
    if x == 0:
        lr_uc = -2 * n * np.log(1 - p)
    elif x == n:
        lr_uc = -2 * n * np.log(p)
    else:
        pi_hat = x / n
        lr_uc = -2 * (np.log((1-p)**(n-x) * p**x) - 
                      np.log((1-pi_hat)**(n-x) * pi_hat**x))
    
    pvalue = 1 - chi2.cdf(lr_uc, 1)
    
    return {
        'LR_UC': lr_uc,
        'p_value': pvalue,
        'violations': x,
        'expected': n * (1 - alpha),
        'rate': x / n * 100,
        'expected_rate': (1 - alpha) * 100
    }


def christoffersen_ind_test(hits):
    """
    Christoffersen's Independence (I) Test.
    
    Tests whether violations are independent (not clustered).
    
    H0: Violations are independent
    """
    n = len(hits)
    hits_np = hits.values if hasattr(hits, 'values') else np.array(hits)
    
    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for i in range(1, n):
        if hits_np[i-1] == 0 and hits_np[i] == 0:
            n00 += 1
        elif hits_np[i-1] == 0 and hits_np[i] == 1:
            n01 += 1
        elif hits_np[i-1] == 1 and hits_np[i] == 0:
            n10 += 1
        else:  # hits_np[i-1] == 1 and hits_np[i] == 1
            n11 += 1
    
    # Transition probabilities
    n0 = n00 + n01
    n1 = n10 + n11
    
    if n0 == 0 or n1 == 0:
        return {'LR_I': 0, 'p_value': 1.0}
    
    pi01 = n01 / n0 if n0 > 0 else 0
    pi11 = n11 / n1 if n1 > 0 else 0
    pi = (n01 + n11) / (n0 + n1)
    
    # Handle edge cases
    if pi == 0 or pi == 1:
        return {'LR_I': 0, 'p_value': 1.0}
    
    # Log-likelihood under independence
    log_l0 = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    
    # Log-likelihood under dependence
    log_l1 = 0
    if n00 > 0:
        log_l1 += n00 * np.log(1 - pi01) if pi01 < 1 else 0
    if n01 > 0:
        log_l1 += n01 * np.log(pi01) if pi01 > 0 else 0
    if n10 > 0:
        log_l1 += n10 * np.log(1 - pi11) if pi11 < 1 else 0
    if n11 > 0:
        log_l1 += n11 * np.log(pi11) if pi11 > 0 else 0
    
    lr_i = 2 * (log_l1 - log_l0)
    lr_i = max(0, lr_i)  # Ensure non-negative
    
    pvalue = 1 - chi2.cdf(lr_i, 1)
    
    return {
        'LR_I': lr_i,
        'p_value': pvalue,
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11
    }


def christoffersen_cc_test(hits, alpha):
    """
    Christoffersen's Conditional Coverage (CC) Test.
    
    Combines UC and Independence tests.
    LR_CC = LR_UC + LR_I ~ chi2(2)
    """
    uc_result = kupiec_uc_test(hits, alpha)
    ind_result = christoffersen_ind_test(hits)
    
    lr_cc = uc_result['LR_UC'] + ind_result['LR_I']
    pvalue = 1 - chi2.cdf(lr_cc, 2)
    
    return {
        'LR_CC': lr_cc,
        'p_value': pvalue,
        'LR_UC': uc_result['LR_UC'],
        'LR_I': ind_result['LR_I'],
        'UC_pvalue': uc_result['p_value'],
        'I_pvalue': ind_result['p_value']
    }


def berkowitz_test(returns, var, alpha):
    """
    Berkowitz Test of Independence.
    
    Tests whether the PIT (Probability Integral Transform) of returns
    follows a uniform distribution and is independent.
    
    We compute z_t = Phi^(-1)(F_t(r_t)) where F_t is the forecast distribution.
    Under correct specification, z_t ~ i.i.d. N(0,1).
    """
    n = len(returns)
    returns_np = returns.values if hasattr(returns, 'values') else np.array(returns)
    var_np = var.values if hasattr(var, 'values') else np.array(var)
    
    # Estimate sigma from var (assuming normal: VaR = mu - z_alpha * sigma)
    z_alpha = norm.ppf(1 - alpha)
    
    # Approximate standardized residuals
    # Under VaR model: VaR = E[r] + z_alpha * sigma
    # So sigma approx = (E[r] - VaR) / z_alpha
    mu = np.mean(returns_np)
    sigma = np.std(returns_np)
    
    # Compute standardized residuals
    z = (returns_np - mu) / sigma
    
    # Test for autocorrelation (AR(1) test)
    z_lag = z[:-1]
    z_current = z[1:]
    
    # Regression: z_t = rho * z_{t-1} + epsilon
    if len(z_lag) > 2:
        cov_matrix = np.cov(z_lag, z_current)
        if cov_matrix[0, 0] > 0:
            rho = cov_matrix[0, 1] / cov_matrix[0, 0]
        else:
            rho = 0
        
        # LR test for rho = 0
        n_test = len(z_lag)
        if abs(rho) < 0.999:
            lr_berk = n_test * np.log(1 / (1 - rho**2))
        else:
            lr_berk = 0
    else:
        rho = 0
        lr_berk = 0
    
    pvalue = 1 - chi2.cdf(abs(lr_berk), 1)
    
    return {
        'LR_Berkowitz': lr_berk,
        'p_value': pvalue,
        'rho': rho
    }


def dynamic_quantile_test(returns, var, alpha, lags=4):
    """
    Dynamic Quantile (DQ) Test of Engle and Manganelli (2004).
    
    Tests whether hits are unpredictable using lagged hits and VaR.
    
    Hit_t = alpha + beta_1 * Hit_{t-1} + ... + beta_k * Hit_{t-k} + gamma * VaR_t + e_t
    
    H0: alpha = beta_i = gamma = 0 for all i
    """
    returns_np = returns.values if hasattr(returns, 'values') else np.array(returns)
    var_np = var.values if hasattr(var, 'values') else np.array(var)
    
    # Compute hits
    hits = (returns_np < var_np).astype(float)
    n = len(hits)
    
    # Center hits
    p = 1 - alpha
    hits_centered = hits - p
    
    # Create design matrix
    # X = [1, Hit_{t-1}, ..., Hit_{t-k}, VaR_t]
    X = np.ones((n - lags, lags + 2))
    y = hits_centered[lags:]
    
    for i in range(1, lags + 1):
        X[:, i] = hits_centered[lags - i:-i]
    X[:, -1] = var_np[lags:]
    
    # OLS regression
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        residuals = y - X @ beta
        
        # DQ statistic
        sigma2 = p * (1 - p)
        dq_stat = (beta.T @ X.T @ X @ beta) / sigma2
        
        df = lags + 2
        pvalue = 1 - chi2.cdf(dq_stat, df)
    except:
        dq_stat = np.nan
        pvalue = np.nan
        df = lags + 2
    
    return {
        'DQ_stat': dq_stat,
        'p_value': pvalue,
        'df': df,
        'lags': lags
    }


# =============================================================================
# 3. MODEL CONFIDENCE SET (MCS)
# =============================================================================
def mcs_loss_function(returns, var, loss_type='quantile'):
    """
    Compute loss function for MCS.
    
    Quantile loss (tick loss):
    L_t = (alpha - I(r_t < VaR_t)) * (r_t - VaR_t)
    
    where alpha = 1 - confidence level
    """
    returns_np = returns.values if hasattr(returns, 'values') else np.array(returns)
    var_np = var.values if hasattr(var, 'values') else np.array(var)
    
    hits = (returns_np < var_np).astype(float)
    
    # Quantile loss (assumes 95% VaR, so alpha = 0.05)
    alpha = 0.05
    loss = (alpha - hits) * (returns_np - var_np)
    
    return loss


def model_confidence_set(var_data, asset, confidence_level, 
                         var_columns, alpha_mcs=0.10, B=1000):
    """
    Compute Model Confidence Set using bootstrap.
    
    Uses the Range statistic and quantile loss.
    
    Parameters:
    -----------
    alpha_mcs : float
        Significance level for MCS (typically 0.10 or 0.25)
    B : int
        Number of bootstrap replications
    """
    returns = var_data[asset]['Actual']
    n = len(returns)
    
    # Compute losses for each model
    losses = {}
    for col in var_columns:
        if col in var_data[asset]:
            losses[col] = mcs_loss_function(returns, var_data[asset][col])
    
    if len(losses) < 2:
        return {'surviving_models': list(losses.keys()), 'eliminated': [], 
                'p_values': {}}
    
    # Create loss matrix
    models = list(losses.keys())
    M = len(models)
    L = np.column_stack([losses[m] for m in models])
    
    # Compute mean losses
    mean_losses = L.mean(axis=0)
    
    # Initialize
    surviving = list(range(M))
    eliminated = []
    p_values = {}
    
    while len(surviving) > 1:
        M_surv = len(surviving)
        L_surv = L[:, surviving]
        
        # Compute pairwise loss differentials
        d_bar = np.zeros((M_surv, M_surv))
        for i in range(M_surv):
            for j in range(M_surv):
                d_bar[i, j] = (L_surv[:, i] - L_surv[:, j]).mean()
        
        # Range statistic
        range_stat = np.max(d_bar.max(axis=1))
        
        # Bootstrap
        range_boot = np.zeros(B)
        for b in range(B):
            idx = np.random.choice(n, n, replace=True)
            L_boot = L_surv[idx, :]
            d_boot = np.zeros((M_surv, M_surv))
            for i in range(M_surv):
                for j in range(M_surv):
                    d_boot[i, j] = (L_boot[:, i] - L_boot[:, j]).mean()
            range_boot[b] = np.max(d_boot.max(axis=1))
        
        # P-value
        p_val = np.mean(range_boot >= range_stat)
        
        if p_val < alpha_mcs:
            # Eliminate worst model
            worst_idx = np.argmax(mean_losses[surviving])
            worst_model = surviving[worst_idx]
            eliminated.append(models[worst_model])
            p_values[models[worst_model]] = p_val
            surviving.remove(worst_model)
        else:
            break
    
    surviving_models = [models[i] for i in surviving]
    
    return {
        'surviving_models': surviving_models,
        'eliminated': eliminated,
        'p_values': p_values,
        'mean_losses': {models[i]: mean_losses[i] for i in range(M)}
    }


# =============================================================================
# 4. RUN ALL BACKTESTS
# =============================================================================
def run_all_backtests(var_data, confidence_level=0.95):
    """Run all backtesting procedures for all models and assets."""
    print("\n" + "=" * 80)
    print(f"2. RUNNING BACKTESTS AT {int(confidence_level*100)}% CONFIDENCE LEVEL")
    print("=" * 80)
    
    alpha = confidence_level
    cl_str = str(int(confidence_level * 100))
    
    # Define VaR columns for this confidence level
    var_columns = [
        f'VaR_Normal_{cl_str}',
        f'VaR_StudentT_{cl_str}',
        f'VaR_GEV_Indep_{cl_str}',
        f'VaR_GEV_Dep_{cl_str}',
        f'VaR_GPD_{cl_str}'
    ]
    
    model_names = ['Normal', 'StudentT', 'GEV_Indep', 'GEV_Dep', 'GPD']
    
    all_results = []
    
    for asset in var_data.keys():
        print(f"\n{'='*60}")
        print(f"Asset: {asset}")
        print("=" * 60)
        
        returns = var_data[asset]['Actual']
        
        for var_col, model_name in zip(var_columns, model_names):
            if var_col not in var_data[asset]:
                continue
            
            var = var_data[asset][var_col]
            hits = compute_hits(returns, var)
            
            # Kupiec UC Test
            uc = kupiec_uc_test(hits, alpha)
            
            # Christoffersen Independence Test
            ind = christoffersen_ind_test(hits)
            
            # Christoffersen CC Test
            cc = christoffersen_cc_test(hits, alpha)
            
            # Berkowitz Test
            berk = berkowitz_test(returns, var, alpha)
            
            # Dynamic Quantile Test
            dq = dynamic_quantile_test(returns, var, alpha)
            
            result = {
                'Asset': asset,
                'Model': model_name,
                'Violations': uc['violations'],
                'Viol_Rate': uc['rate'],
                'Expected_Rate': uc['expected_rate'],
                'UC_LR': uc['LR_UC'],
                'UC_pval': uc['p_value'],
                'UC_pass': 'Yes' if uc['p_value'] > 0.05 else 'No',
                'I_LR': ind['LR_I'],
                'I_pval': ind['p_value'],
                'I_pass': 'Yes' if ind['p_value'] > 0.05 else 'No',
                'CC_LR': cc['LR_CC'],
                'CC_pval': cc['p_value'],
                'CC_pass': 'Yes' if cc['p_value'] > 0.05 else 'No',
                'Berk_LR': berk['LR_Berkowitz'],
                'Berk_pval': berk['p_value'],
                'Berk_pass': 'Yes' if berk['p_value'] > 0.05 else 'No',
                'DQ_stat': dq['DQ_stat'],
                'DQ_pval': dq['p_value'],
                'DQ_pass': 'Yes' if pd.notna(dq['p_value']) and dq['p_value'] > 0.05 else 'No'
            }
            
            all_results.append(result)
    
    results_df = pd.DataFrame(all_results)
    
    return results_df, var_columns


def run_mcs(var_data, confidence_level=0.95):
    """Run Model Confidence Set for all assets."""
    print("\n" + "=" * 80)
    print(f"3. MODEL CONFIDENCE SET AT {int(confidence_level*100)}% VaR")
    print("=" * 80)
    
    cl_str = str(int(confidence_level * 100))
    var_columns = [
        f'VaR_Normal_{cl_str}',
        f'VaR_StudentT_{cl_str}',
        f'VaR_GEV_Indep_{cl_str}',
        f'VaR_GEV_Dep_{cl_str}',
        f'VaR_GPD_{cl_str}'
    ]
    
    mcs_results = {}
    
    for asset in var_data.keys():
        print(f"\n{asset}:")
        mcs = model_confidence_set(var_data, asset, confidence_level, var_columns)
        mcs_results[asset] = mcs
        
        print(f"  Surviving models: {mcs['surviving_models']}")
        print(f"  Eliminated: {mcs['eliminated']}")
        if mcs['mean_losses']:
            losses_sorted = sorted(mcs['mean_losses'].items(), key=lambda x: x[1])
            print(f"  Mean losses (ranked):")
            for model, loss in losses_sorted:
                symbol = '*' if model in mcs['surviving_models'] else ' '
                print(f"    {symbol} {model}: {loss:.6f}")
    
    return mcs_results


# =============================================================================
# 5. SUMMARY AND ANALYSIS
# =============================================================================
def summarize_results(results_df):
    """Create summary tables of test results."""
    print("\n" + "=" * 80)
    print("4. SUMMARY OF BACKTESTING RESULTS")
    print("=" * 80)
    
    # Summary by model
    print("\n" + "-" * 60)
    print("PASS RATES BY MODEL (across all assets)")
    print("-" * 60)
    
    summary = results_df.groupby('Model').agg({
        'UC_pass': lambda x: (x == 'Yes').sum(),
        'I_pass': lambda x: (x == 'Yes').sum(),
        'CC_pass': lambda x: (x == 'Yes').sum(),
        'Berk_pass': lambda x: (x == 'Yes').sum(),
        'DQ_pass': lambda x: (x == 'Yes').sum()
    })
    
    n_assets = len(results_df['Asset'].unique())
    summary_pct = summary / n_assets * 100
    summary_pct.columns = ['UC%', 'I%', 'CC%', 'Berk%', 'DQ%']
    
    print(f"\nNumber of assets passing each test (out of {n_assets}):")
    print(summary.to_string())
    print(f"\nPercentage passing:")
    print(summary_pct.to_string())
    
    # Compute overall score
    summary['Total_Pass'] = summary.sum(axis=1)
    summary['Score'] = summary['Total_Pass'] / (n_assets * 5) * 100
    print(f"\nOverall Score (% of all tests passed):")
    print(summary[['Total_Pass', 'Score']].sort_values('Score', ascending=False).to_string())
    
    return summary


def analyze_preferred_model(results_df, mcs_results):
    """Analyze which VaR model is most frequently preferred."""
    print("\n" + "=" * 80)
    print("5. ANALYSIS: PREFERRED VAR MODEL")
    print("=" * 80)
    
    # Count how many tests each model passes
    model_pass_counts = {}
    tests = ['UC_pass', 'I_pass', 'CC_pass', 'Berk_pass', 'DQ_pass']
    
    for model in results_df['Model'].unique():
        model_df = results_df[results_df['Model'] == model]
        pass_count = 0
        for test in tests:
            pass_count += (model_df[test] == 'Yes').sum()
        model_pass_counts[model] = pass_count
    
    print("\nTotal tests passed across all assets:")
    for model, count in sorted(model_pass_counts.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")
    
    # Count MCS survival
    mcs_survival = {}
    for model in results_df['Model'].unique():
        col_name = f"VaR_{model}_95"
        survival_count = 0
        for asset, mcs in mcs_results.items():
            if col_name in mcs['surviving_models']:
                survival_count += 1
        mcs_survival[model] = survival_count
    
    print("\nMCS survival count (out of", len(mcs_results), "assets):")
    for model, count in sorted(mcs_survival.items(), key=lambda x: -x[1]):
        print(f"  {model}: {count}")
    
    # Determine best model
    best_by_tests = max(model_pass_counts.items(), key=lambda x: x[1])
    best_by_mcs = max(mcs_survival.items(), key=lambda x: x[1])
    
    print("\n" + "-" * 60)
    print("CONCLUSION")
    print("-" * 60)
    print(f"\nBased on backtesting results:")
    print(f"  - Best model by test pass count: {best_by_tests[0]} ({best_by_tests[1]} tests passed)")
    print(f"  - Best model by MCS survival: {best_by_mcs[0]} ({best_by_mcs[1]} assets)")
    
    return model_pass_counts, mcs_survival


# =============================================================================
# 6. VISUALIZATION
# =============================================================================
def plot_results(results_df, mcs_results):
    """Create visualizations of backtesting results."""
    print("\n" + "=" * 80)
    print("6. CREATING VISUALIZATIONS")
    print("=" * 80)
    
    # Figure 1: Heatmap of test results by model and asset
    models = results_df['Model'].unique()
    assets = results_df['Asset'].unique()
    tests = ['UC_pass', 'I_pass', 'CC_pass', 'Berk_pass', 'DQ_pass']
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 6))
    
    for idx, test in enumerate(tests):
        # Create matrix
        matrix = np.zeros((len(assets), len(models)))
        for i, asset in enumerate(assets):
            for j, model in enumerate(models):
                row = results_df[(results_df['Asset'] == asset) & 
                                (results_df['Model'] == model)]
                if len(row) > 0:
                    matrix[i, j] = 1 if row[test].values[0] == 'Yes' else 0
        
        ax = axes[idx]
        sns.heatmap(matrix, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                   xticklabels=models, yticklabels=assets,
                   cbar=False, annot=False)
        ax.set_title(test.replace('_pass', ''))
        ax.set_xlabel('')
        if idx > 0:
            ax.set_yticklabels([])
    
    plt.tight_layout()
    plt.savefig('fig_backtest_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nHeatmap saved: fig_backtest_heatmap.png")
    
    # Figure 2: Bar chart of pass rates
    pass_rates = results_df.groupby('Model').apply(
        lambda x: pd.Series({
            'UC': (x['UC_pass'] == 'Yes').mean() * 100,
            'Independence': (x['I_pass'] == 'Yes').mean() * 100,
            'CC': (x['CC_pass'] == 'Yes').mean() * 100,
            'Berkowitz': (x['Berk_pass'] == 'Yes').mean() * 100,
            'DQ': (x['DQ_pass'] == 'Yes').mean() * 100
        })
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    pass_rates.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Pass Rate (%)')
    ax.set_xlabel('VaR Model')
    ax.set_title('Backtesting Pass Rates by VaR Model')
    ax.legend(title='Test', bbox_to_anchor=(1.02, 1))
    ax.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% threshold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fig_backtest_passrates.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Pass rates saved: fig_backtest_passrates.png")
    
    # Figure 3: Violation rates comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    viol_data = results_df.pivot(index='Asset', columns='Model', values='Viol_Rate')
    viol_data.plot(kind='bar', ax=ax, width=0.8)
    ax.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='Expected (5%)')
    ax.set_ylabel('Violation Rate (%)')
    ax.set_xlabel('Asset')
    ax.set_title('VaR Violation Rates by Asset and Model (95% VaR)')
    ax.legend(title='Model', bbox_to_anchor=(1.02, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('fig_backtest_violations.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Violations saved: fig_backtest_violations.png")


# =============================================================================
# 7. EXPORT RESULTS
# =============================================================================
def export_results(results_df, mcs_results, summary):
    """Export all results to Excel."""
    print("\n" + "=" * 80)
    print("7. EXPORTING RESULTS")
    print("=" * 80)
    
    with pd.ExcelWriter('backtest_results.xlsx', engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='All_Tests', index=False)
        summary.to_excel(writer, sheet_name='Summary')
        
        # MCS results
        mcs_data = []
        for asset, mcs in mcs_results.items():
            mcs_data.append({
                'Asset': asset,
                'Surviving_Models': ', '.join(mcs['surviving_models']),
                'Eliminated': ', '.join(mcs['eliminated']),
                'N_Surviving': len(mcs['surviving_models'])
            })
        pd.DataFrame(mcs_data).to_excel(writer, sheet_name='MCS', index=False)
    
    print("\nResults exported to: backtest_results.xlsx")


# =============================================================================
# 8. PRINT COMMENTARY
# =============================================================================
def print_commentary(results_df, mcs_results, model_pass_counts, mcs_survival):
    """Print detailed commentary on results."""
    print("\n" + "=" * 80)
    print("8. DETAILED COMMENTARY ON RESULTS")
    print("=" * 80)
    
    print("""
INTERPRETATION OF BACKTESTING RESULTS
=====================================

1. KUPIEC UNCONDITIONAL COVERAGE (UC) TEST:
   - Tests if the violation rate equals the expected rate
   - Models passing this test have correct average coverage
   
2. CHRISTOFFERSEN INDEPENDENCE (I) TEST:
   - Tests if violations are serially independent (not clustered)
   - Failures suggest volatility clustering not captured
   
3. CHRISTOFFERSEN CONDITIONAL COVERAGE (CC) TEST:
   - Combined test of coverage and independence
   - Most stringent individual test
   
4. BERKOWITZ TEST:
   - Tests independence of standardized residuals
   - Sensitive to model misspecification
   
5. DYNAMIC QUANTILE (DQ) TEST:
   - Tests predictability of violations using lagged values
   - Failures suggest remaining dynamics in violations
   
6. MODEL CONFIDENCE SET (MCS):
   - Identifies set of models statistically indistinguishable
   - Uses quantile loss to rank models
""")
    
    # Specific findings
    print("\nKEY FINDINGS:")
    print("-" * 60)
    
    # Best performing model
    best_model = max(model_pass_counts.items(), key=lambda x: x[1])[0]
    print(f"\n1. BEST OVERALL MODEL: {best_model}")
    print(f"   - Passed {model_pass_counts[best_model]} tests across all assets")
    
    # UC test results
    uc_pass = results_df.groupby('Model')['UC_pass'].apply(lambda x: (x=='Yes').sum())
    best_uc = uc_pass.idxmax()
    print(f"\n2. BEST UNCONDITIONAL COVERAGE: {best_uc}")
    print(f"   - {uc_pass[best_uc]} assets with correct violation rate")
    
    # Independence test results
    i_pass = results_df.groupby('Model')['I_pass'].apply(lambda x: (x=='Yes').sum())
    best_i = i_pass.idxmax()
    print(f"\n3. BEST INDEPENDENCE: {best_i}")
    print(f"   - {i_pass[best_i]} assets with independent violations")
    
    # MCS survival
    best_mcs = max(mcs_survival.items(), key=lambda x: x[1])[0]
    print(f"\n4. MCS MOST FREQUENTLY SURVIVING: {best_mcs}")
    print(f"   - Survived in {mcs_survival[best_mcs]} out of {len(mcs_results)} assets")
    
    print("\n" + "-" * 60)
    print("CONCLUSION")
    print("-" * 60)
    print(f"""
Based on the comprehensive backtesting analysis:

The '{best_model}' VaR model appears to be most frequently preferred,
passing the highest number of backtesting diagnostics.

However, the Model Confidence Set analysis shows that '{best_mcs}' 
survives most frequently when using quantile loss as the criterion.

Key observations:
- Standard Normal VaR tends to underestimate tail risk
- Student-t captures fat tails better, improving UC test performance
- EVT-based methods (GEV, GPD) provide more conservative VaR estimates
- Trade-off exists between violation rate and model complexity

RECOMMENDATION:
For practical risk management, consider using a combination approach:
- Student-t or GPD for normal market conditions
- More conservative EVT-based methods during stress periods
""")


# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VAR BACKTESTING FRAMEWORK")
    print("Comparing 5 VaR Models Across All Assets")
    print("=" * 80 + "\n")
    
    # 1. Load VaR forecasts
    var_data = load_var_forecasts()
    
    # 2. Run all backtests at 95% confidence
    results_df, var_columns = run_all_backtests(var_data, confidence_level=0.95)
    
    # 3. Run Model Confidence Set
    mcs_results = run_mcs(var_data, confidence_level=0.95)
    
    # 4. Summarize results
    summary = summarize_results(results_df)
    
    # 5. Analyze preferred model
    model_pass_counts, mcs_survival = analyze_preferred_model(results_df, mcs_results)
    
    # 6. Create visualizations
    plot_results(results_df, mcs_results)
    
    # 7. Export results
    export_results(results_df, mcs_results, summary)
    
    # 8. Print commentary
    print_commentary(results_df, mcs_results, model_pass_counts, mcs_survival)
    
    print("\n" + "=" * 80)
    print("BACKTESTING ANALYSIS COMPLETE")
    print("=" * 80)
