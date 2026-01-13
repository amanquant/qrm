import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="QRM Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .code-snippet {
        background-color: #f4f4f4;
        border-radius: 5px;
        padding: 1rem;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)


def load_image(filename):
    """Load and display image if exists."""
    if os.path.exists(filename):
        return filename
    return None


def show_code_snippet(code, language="python"):
    """Display code snippet with syntax highlighting."""
    st.code(code, language=language)


# Sidebar navigation
st.sidebar.markdown("# 📊 QRM Analysis")
st.sidebar.markdown("---")

parts = {
    "🏠 Overview": "overview",
    "📈 Part 1: Descriptive Analysis": "part1",
    "📉 Part 2: GARCH Modeling": "part2",
    "⚠️ Part 3: VaR Normal & Student-t": "part3",
    "🎯 Part 4: VaR with GEV": "part4",
    "📊 Part 5: VaR with GPD": "part5",
    "✅ Part 6: Backtesting": "part6",
    "💰 Part 7: Expected Shortfall": "part7"
}

selected = st.sidebar.radio("Select Section", list(parts.keys()))
selected_part = parts[selected]

st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Data")
st.sidebar.info("**Source:** Manzi.xlsx\n\n**Assets:** Walmart, Costco, HomeDepot, Pepsico, TJX, Lowes\n\n**Period:** 2005-2025")


# =============================================================================
# OVERVIEW
# =============================================================================
if selected_part == "overview":
    st.markdown('<div class="main-header">📊 Quantitative Risk Management Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the QRM Analysis Dashboard! This app showcases a comprehensive financial risk analysis 
    covering **6 U.S. retail sector stocks** over **20 years of daily data**.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📅 Time Period", "2005-2025", "20 years")
    with col2:
        st.metric("📊 Assets Analyzed", "6 Stocks", "Retail Sector")
    with col3:
        st.metric("📈 Observations", "5,478", "Daily returns")
    
    st.markdown("---")
    st.markdown("### 🎯 Key Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Best VaR Models:**
        - GARCH + Normal (96.7% pass rate)
        - GARCH + GPD (96.7% pass rate)
        """)
        
        st.warning("""
        **Riskiest Asset:** Lowe's
        - 95% ES: -4.29%
        - 99% ES: -7.21%
        """)
    
    with col2:
        st.info("""
        **Stylized Facts Confirmed:**
        - Non-normality ✓
        - Volatility clustering ✓
        - Fat tails ✓
        """)
        
        st.success("""
        **Safest Asset:** Pepsico
        - 95% ES: -2.63%
        - 99% ES: -4.54%
        """)
    
    st.markdown("---")
    st.markdown("### 📑 Analysis Parts")
    
    parts_info = {
        "Part 1": "Descriptive statistics, normality tests, ARCH effects",
        "Part 2": "GARCH(1,1) modeling with Normal distribution",
        "Part 3": "VaR with Normal and Student-t distributions",
        "Part 4": "VaR with GEV (independence and dependence)",
        "Part 5": "VaR with GPD (Peaks Over Threshold)",
        "Part 6": "Comprehensive backtesting framework",
        "Part 7": "Expected Shortfall analysis"
    }
    
    for part, desc in parts_info.items():
        st.markdown(f"**{part}:** {desc}")


# =============================================================================
# PART 1: DESCRIPTIVE ANALYSIS
# =============================================================================
elif selected_part == "part1":
    st.markdown("# 📈 Part 1: Descriptive Analysis")
    
    st.markdown("""
    > *Compute logarithmic returns and perform descriptive analysis, highlighting serial dependence, 
    asymmetry, deviations from normality, volatility clustering, and extreme events.*
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### Descriptive Statistics")
        
        stats_data = {
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Mean': [0.041, 0.074, 0.068, 0.043, 0.068, 0.057],
            'Std Dev': [1.42, 1.61, 1.76, 1.17, 1.70, 1.95],
            'Skewness': [0.006, -0.277, -0.317, -0.436, -0.387, -0.580],
            'Kurtosis': [12.87, 6.95, 5.42, 17.42, 5.61, 16.01],
            'Min': [-11.8, -12.4, -14.5, -15.0, -14.9, -20.2],
            'Max': [11.9, 9.5, 11.1, 9.8, 10.0, 15.2]
        }
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
        
        st.markdown("### Key Findings")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Serial Dependence:** ✅ Significant in all assets")
            st.markdown("**Asymmetry:** ✅ 5/6 assets negatively skewed")
            st.markdown("**Normality:** ❌ All assets reject normality")
        
        with col2:
            st.markdown("**ARCH Effects:** ✅ Strong volatility clustering")
            st.markdown("**Extreme Events:** 3-6x more than expected under normality")
            st.markdown("**Time-Varying Variance:** ✅ Non-stationary")
    
    with tab2:
        st.markdown("### Generated Charts")
        
        images = [
            ('fig1_log_returns_timeseries.png', 'Log Returns Time Series'),
            ('fig2_return_distributions.png', 'Return Distributions'),
            ('fig3_acf_plots.png', 'ACF Plots'),
            ('fig4_rolling_correlations.png', 'Rolling Correlations'),
            ('fig5_correlation_heatmap.png', 'Correlation Heatmap'),
            ('fig6_rolling_volatility.png', 'Rolling Volatility')
        ]
        
        for img, title in images:
            if os.path.exists(img):
                st.markdown(f"**{title}**")
                st.image(img, use_container_width=True)
            else:
                st.info(f"Chart '{img}' not found. Run part1.py to generate.")
    
    with tab3:
        st.markdown("### Python Code (`part1.py`)")
        code = '''
# Log returns computation
def compute_log_returns(prices_df):
    log_returns = np.log(prices_df / prices_df.shift(1))
    log_returns.dropna(inplace=True)
    log_returns_scaled = log_returns * 100
    return log_returns, log_returns_scaled

# Normality tests
def test_normality(log_returns):
    for asset in log_returns.columns:
        jb_stat, jb_pval = stats.jarque_bera(log_returns[asset])
        print(f"{asset}: JB={jb_stat:.2f}, p={jb_pval:.4f}")

# ARCH-LM test for volatility clustering
from statsmodels.stats.diagnostic import het_arch
for asset in assets:
    lm_stat, lm_pval, _, _ = het_arch(returns_demeaned[asset])
    print(f"{asset}: ARCH effect p-value = {lm_pval:.4f}")
'''
        st.code(code, language="python")


# =============================================================================
# PART 2: GARCH MODELING
# =============================================================================
elif selected_part == "part2":
    st.markdown("# 📉 Part 2: GARCH Modeling")
    
    st.markdown("""
    > *Fit univariate GARCH model using Normal density. Comment on statistical significance. 
    Plot conditional variance.*
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### GARCH(1,1) Parameter Estimates")
        
        garch_data = {
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'mu': [0.041, 0.090, 0.084, 0.050, 0.082, 0.075],
            'omega': [0.037, 0.057, 0.054, 0.045, 0.056, 0.125],
            'alpha': [0.051, 0.068, 0.088, 0.086, 0.076, 0.091],
            'beta': [0.925, 0.902, 0.889, 0.870, 0.902, 0.871],
            'Persistence': [0.977, 0.970, 0.977, 0.955, 0.978, 0.963]
        }
        st.dataframe(pd.DataFrame(garch_data), use_container_width=True)
        
        st.markdown("### Interpretation")
        st.info("""
        - **High Persistence (α+β ≈ 0.97):** Volatility shocks decay slowly
        - **Beta significant in all assets:** Strong GARCH effect
        - **All models stationary:** α+β < 1
        """)
    
    with tab2:
        images = [
            ('fig_garch_conditional_volatility_all.png', 'Conditional Volatility - All Assets'),
            ('fig_garch_detailed_Walmart.png', 'Detailed GARCH Analysis - Walmart'),
            ('fig_garch_volatility_comparison.png', 'Volatility Comparison')
        ]
        
        for img, title in images:
            if os.path.exists(img):
                st.markdown(f"**{title}**")
                st.image(img, use_container_width=True)
            else:
                st.info(f"Chart '{img}' not found. Run part2.py to generate.")
    
    with tab3:
        code = '''
from arch import arch_model

# Fit GARCH(1,1) with Normal distribution
model = arch_model(returns, mean='Constant', vol='GARCH', 
                   p=1, q=1, dist='Normal')
result = model.fit(disp='off')

# Extract parameters
print(result.summary())
print(f"Persistence: {result.params['alpha[1]'] + result.params['beta[1]']:.4f}")

# Get conditional volatility
cond_vol = result.conditional_volatility
'''
        st.code(code, language="python")


# =============================================================================
# PART 3: VaR NORMAL & STUDENT-T
# =============================================================================
elif selected_part == "part3":
    st.markdown("# ⚠️ Part 3: VaR with Normal & Student-t")
    
    st.markdown("""
    > *Compute VaR under Normality (Case 1) and with Student-t distribution (Case 2).*
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### VaR Backtesting Results (95% Confidence)")
        
        var_data = {
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Normal Violations': [10, 14, 13, 20, 7, 10],
            'Normal Rate': ['3.85%', '5.38%', '5.00%', '7.69%', '2.69%', '3.85%'],
            'Student-t Violations': [2, 4, 4, 5, 0, 1],
            'Student-t Rate': ['0.77%', '1.54%', '1.54%', '1.92%', '0.00%', '0.38%']
        }
        st.dataframe(pd.DataFrame(var_data), use_container_width=True)
        
        st.warning("**Expected Rate: 5%** - Normal is closer to expected; Student-t is too conservative.")
    
    with tab2:
        images = [
            ('fig_var_forecasts_95.png', 'VaR Forecasts (95%)'),
            ('fig_var_comparison_Walmart.png', 'VaR Comparison - Walmart'),
            ('fig_forecast_variance.png', 'Forecast Variance')
        ]
        
        for img, title in images:
            if os.path.exists(img):
                st.markdown(f"**{title}**")
                st.image(img, use_container_width=True)
    
    with tab3:
        code = '''
from scipy.stats import norm, t as student_t

# VaR Case 1: Normal
z_alpha = norm.ppf(0.05)  # 95% VaR
var_normal = mu + z_alpha * sigma

# VaR Case 2: Student-t
# Fit GARCH with t-distribution
model_t = arch_model(data, dist='t')
result_t = model_t.fit()
nu = result_t.params['nu']  # degrees of freedom

t_alpha = student_t.ppf(0.05, df=nu)
var_t = mu + t_alpha * sigma * np.sqrt((nu-2)/nu)
'''
        st.code(code, language="python")


# =============================================================================
# PART 4: VaR WITH GEV
# =============================================================================
elif selected_part == "part4":
    st.markdown("# 🎯 Part 4: VaR with GEV")
    
    st.markdown("""
    > *Fit GEV on standardized residuals with independence (Case 3) and dependence (Case 4).*
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### GEV Parameters")
        
        gev_data = {
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Extremal Index': [0.931, 0.950, 0.939, 0.962, 0.962, 0.946],
            'xi (shape)': [0.202, 0.242, 0.054, 0.130, 0.085, 0.178],
            'mu (location)': [1.466, 1.496, 1.662, 1.624, 1.643, 1.549],
            'sigma (scale)': [0.663, 0.609, 0.644, 0.620, 0.541, 0.636]
        }
        st.dataframe(pd.DataFrame(gev_data), use_container_width=True)
        
        st.info("All ξ > 0 indicates **heavy-tailed (Fréchet-type)** distributions.")
    
    with tab2:
        images = [
            ('fig_gev_qq_plots.png', 'Q-Q Plots'),
            ('fig_gev_block_maxima.png', 'Block Maxima with GEV Fit'),
            ('fig_gev_var_comparison.png', 'GEV VaR Comparison')
        ]
        
        for img, title in images:
            if os.path.exists(img):
                st.markdown(f"**{title}**")
                st.image(img, use_container_width=True)
    
    with tab3:
        code = '''
from scipy.stats import genextreme

# Extract block maxima (21-day blocks)
def extract_block_minima(data, block_size=21):
    neg_data = -data.values
    n_blocks = len(neg_data) // block_size
    block_maxima = [np.max(neg_data[i*block_size:(i+1)*block_size]) 
                    for i in range(n_blocks)]
    return np.array(block_maxima)

# Fit GEV
params = genextreme.fit(block_maxima)
xi = -params[0]  # shape (scipy uses c = -xi)
mu = params[1]   # location
sigma = params[2] # scale
'''
        st.code(code, language="python")


# =============================================================================
# PART 5: VaR WITH GPD
# =============================================================================
elif selected_part == "part5":
    st.markdown("# 📊 Part 5: VaR with GPD")
    
    st.markdown("""
    > *Implement GARCH + GPD (Peaks Over Threshold) as Case 5.*
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### GPD Parameters")
        
        gpd_data = {
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Threshold': [1.066, 1.134, 1.208, 1.180, 1.209, 1.160],
            'Exceedances': [522, 522, 522, 522, 522, 522],
            'xi (shape)': [0.212, 0.208, 0.050, 0.089, 0.073, 0.164],
            'sigma (scale)': [0.546, 0.528, 0.624, 0.613, 0.549, 0.556]
        }
        st.dataframe(pd.DataFrame(gpd_data), use_container_width=True)
        
        st.markdown("### Backtesting (95% VaR)")
        backtest = {
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Violations': [16, 16, 13, 21, 8, 11],
            'Rate': ['6.15%', '6.15%', '5.00%', '8.08%', '3.08%', '4.23%'],
            'POF p-value': [0.409, 0.409, 1.000, 0.036, 0.127, 0.559]
        }
        st.dataframe(pd.DataFrame(backtest), use_container_width=True)
    
    with tab2:
        images = [
            ('fig_gpd_exceedances.png', 'Exceedances with GPD Fit'),
            ('fig_gpd_var_forecasts.png', 'GPD VaR Forecasts'),
            ('fig_gpd_mean_excess.png', 'Mean Excess Plots')
        ]
        
        for img, title in images:
            if os.path.exists(img):
                st.markdown(f"**{title}**")
                st.image(img, use_container_width=True)
    
    with tab3:
        code = '''
from scipy.stats import genpareto

# Select threshold (90th percentile)
threshold = np.quantile(-z, 0.90)

# Extract exceedances
exceedances = (-z)[(-z) > threshold] - threshold

# Fit GPD
params = genpareto.fit(exceedances, floc=0)
xi = params[0]     # shape
sigma = params[2]  # scale

# GPD VaR quantile
def gpd_var_quantile(xi, sigma, threshold, p, n, n_exceed):
    Fu = n_exceed / n
    return threshold + (sigma/xi) * ((Fu/p)**xi - 1)
'''
        st.code(code, language="python")


# =============================================================================
# PART 6: BACKTESTING
# =============================================================================
elif selected_part == "part6":
    st.markdown("# ✅ Part 6: Backtesting Evaluation")
    
    st.markdown("""
    > *Perform backtesting with Kupiec, Christoffersen, Berkowitz, DQ, and MCS tests.*
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### Overall Backtesting Results (95% VaR)")
        
        results = {
            'Model': ['Normal', 'GPD', 'Student-t', 'GEV (Dep)', 'GEV (Indep)'],
            'UC Pass': ['6/6', '5/6', '0/6', '0/6', '0/6'],
            'I Pass': ['5/6', '6/6', '6/6', '6/6', '6/6'],
            'CC Pass': ['6/6', '6/6', '0/6', '0/6', '0/6'],
            'Berk Pass': ['6/6', '6/6', '6/6', '6/6', '6/6'],
            'DQ Pass': ['6/6', '6/6', '5/6', '0/6', '0/6'],
            'Score': ['96.7%', '96.7%', '56.7%', '40.0%', '40.0%']
        }
        st.dataframe(pd.DataFrame(results), use_container_width=True)
        
        st.success("**Preferred Models:** GARCH-Normal and GARCH-GPD (tied at 96.7%)")
    
    with tab2:
        images = [
            ('fig_backtest_heatmap.png', 'Backtesting Heatmap'),
            ('fig_backtest_passrates.png', 'Pass Rates by Model'),
            ('fig_backtest_violations.png', 'Violation Rates')
        ]
        
        for img, title in images:
            if os.path.exists(img):
                st.markdown(f"**{title}**")
                st.image(img, use_container_width=True)
    
    with tab3:
        code = '''
# Kupiec UC Test
def kupiec_uc_test(hits, alpha):
    n, x = len(hits), hits.sum()
    p = 1 - alpha
    lr_uc = -2 * (np.log((1-p)**(n-x) * p**x) - 
                  np.log((1-x/n)**(n-x) * (x/n)**x))
    return 1 - chi2.cdf(lr_uc, 1)

# Christoffersen Independence Test
def christoffersen_ind_test(hits):
    # Count transitions: n00, n01, n10, n11
    # LR_I = 2*(log_l1 - log_l0)

# Dynamic Quantile Test
def dq_test(returns, var, alpha, lags=4):
    # Regress hits on lagged hits and VaR
    # DQ_stat = beta'X'X*beta / sigma2
'''
        st.code(code, language="python")


# =============================================================================
# PART 7: EXPECTED SHORTFALL
# =============================================================================
elif selected_part == "part7":
    st.markdown("# 💰 Part 7: Expected Shortfall")
    
    st.markdown("""
    > *Compute Expected Shortfall and compare ES among assets.*
    """)
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### Expected Shortfall Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**95% Confidence Level**")
            es_95 = {
                'Asset': ['Lowes', 'HomeDepot', 'TJX', 'Costco', 'Walmart', 'Pepsico'],
                'VaR': ['-2.73%', '-2.42%', '-2.31%', '-2.01%', '-1.79%', '-1.62%'],
                'ES': ['-4.29%', '-3.73%', '-3.68%', '-3.19%', '-2.86%', '-2.63%'],
                'Rank': ['1 (Riskiest)', '2', '3', '4', '5', '6 (Safest)']
            }
            st.dataframe(pd.DataFrame(es_95), use_container_width=True)
        
        with col2:
            st.markdown("**99% Confidence Level**")
            es_99 = {
                'Asset': ['Lowes', 'TJX', 'HomeDepot', 'Costco', 'Walmart', 'Pepsico'],
                'VaR': ['-5.06%', '-4.15%', '-4.43%', '-3.71%', '-3.23%', '-3.00%'],
                'ES': ['-7.21%', '-6.23%', '-6.18%', '-5.62%', '-5.09%', '-4.54%']
            }
            st.dataframe(pd.DataFrame(es_99), use_container_width=True)
        
        st.info("**ES/VaR Ratio ≈ 1.5-1.6:** ES is ~50-60% worse than VaR in extreme scenarios.")
    
    with tab2:
        images = [
            ('fig_es_var_comparison.png', 'VaR vs ES Comparison'),
            ('fig_es_var_ratio.png', 'ES/VaR Ratio'),
            ('fig_es_distributions.png', 'Return Distributions with VaR/ES'),
            ('fig_es_rolling.png', 'Rolling ES'),
            ('fig_es_boxplot.png', 'Tail Returns Box Plot'),
            ('fig_es_radar.png', 'Risk Radar Chart')
        ]
        
        for img, title in images:
            if os.path.exists(img):
                st.markdown(f"**{title}**")
                st.image(img, use_container_width=True)
    
    with tab3:
        code = '''
# Historical Simulation ES
def compute_historical_es(returns, confidence_levels=[0.95, 0.99]):
    for cl in confidence_levels:
        alpha = 1 - cl
        var = np.percentile(returns, alpha * 100)
        tail_returns = returns[returns <= var]
        es = np.mean(tail_returns)
        print(f"{cl*100}% VaR: {var:.4f}, ES: {es:.4f}")
'''
        st.code(code, language="python")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Run Analysis")
st.sidebar.markdown("""
```bash
python part1.py  # Descriptive
python part2.py  # GARCH
python part3.py  # VaR 1-2
python part4.py  # VaR 3-4
python part5.py  # VaR 5
python part6.py  # Backtesting
python part7.py  # ES
```
""")

st.sidebar.markdown("---")
st.sidebar.caption("QRM Analysis Dashboard v1.0")
