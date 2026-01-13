"""
QRM Analysis Showcase - Streamlit App
======================================
Interactive dashboard with dynamically generated charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

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
</style>
""", unsafe_allow_html=True)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load and prepare data."""
    from pathlib import Path
    
    # Get the directory where the script is located
    script_dir = Path(__file__).parent
    data_file = script_dir / 'Manzi.xlsx'
    
    # Also try current working directory
    if not data_file.exists():
        data_file = Path('Manzi.xlsx')
    
    asset_names = ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes']
    
    try:
        df = pd.read_excel(data_file, skiprows=2)
        df.columns = ['Date'] + asset_names
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        # Compute log returns
        log_returns = np.log(df / df.shift(1)) * 100
        log_returns.dropna(inplace=True)
        
        return df, log_returns, asset_names
    except Exception as e:
        st.warning(f"Data file not found. Using sample data for demonstration. Error: {e}")
        
        # Create sample data for demonstration
        np.random.seed(42)
        dates = pd.date_range('2005-01-03', '2025-12-31', freq='B')
        sample_returns = pd.DataFrame({
            asset: np.random.normal(0.05, 1.5, len(dates)) 
            for asset in asset_names
        }, index=dates)
        
        return None, sample_returns, asset_names


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
st.sidebar.info("**Assets:** Walmart, Costco, HomeDepot, Pepsico, TJX, Lowes\n\n**Period:** 2005-2025")


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
        st.success("**Best VaR Models:**\n- GARCH + Normal (96.7%)\n- GARCH + GPD (96.7%)")
        st.warning("**Riskiest Asset:** Lowe's\n- 95% ES: -4.29%")
    
    with col2:
        st.info("**Stylized Facts:**\n- Non-normality ✓\n- Volatility clustering ✓\n- Fat tails ✓")
        st.success("**Safest Asset:** Pepsico\n- 95% ES: -2.63%")


# =============================================================================
# PART 1: DESCRIPTIVE ANALYSIS
# =============================================================================
elif selected_part == "part1":
    st.markdown("# 📈 Part 1: Descriptive Analysis")
    
    prices_df, log_returns, asset_names = load_data()
    
    if log_returns is not None:
        tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
        
        with tab1:
            st.markdown("### Descriptive Statistics")
            stats = log_returns.describe().T
            stats['Skewness'] = log_returns.skew()
            stats['Kurtosis'] = log_returns.kurtosis()
            st.dataframe(stats[['mean', 'std', 'min', 'max', 'Skewness', 'Kurtosis']], use_container_width=True)
            
            st.markdown("### Key Findings")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Serial Dependence:** ✅ Significant")
                st.markdown("**Asymmetry:** ✅ 5/6 negatively skewed")
                st.markdown("**Normality:** ❌ All reject normality")
            with col2:
                st.markdown("**ARCH Effects:** ✅ Strong clustering")
                st.markdown("**Extreme Events:** 3-6x more than expected")
        
        with tab2:
            st.markdown("### Log Returns Time Series")
            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            axes = axes.flatten()
            for idx, asset in enumerate(asset_names):
                axes[idx].plot(log_returns.index, log_returns[asset], linewidth=0.5)
                axes[idx].set_title(asset)
                axes[idx].set_ylabel('Return (%)')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### Return Distributions")
            fig, axes = plt.subplots(2, 3, figsize=(14, 8))
            axes = axes.flatten()
            for idx, asset in enumerate(asset_names):
                axes[idx].hist(log_returns[asset], bins=50, density=True, alpha=0.7, edgecolor='black')
                axes[idx].set_title(asset)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
            st.pyplot(fig)
        
        with tab3:
            st.code('''
# Log returns computation
log_returns = np.log(prices / prices.shift(1)) * 100

# Descriptive statistics
stats = log_returns.describe().T
stats['Skewness'] = log_returns.skew()
stats['Kurtosis'] = log_returns.kurtosis()

# Normality test
from scipy.stats import jarque_bera
jb_stat, jb_pval = jarque_bera(returns)
            ''', language="python")


# =============================================================================
# PART 2: GARCH MODELING
# =============================================================================
elif selected_part == "part2":
    st.markdown("# 📉 Part 2: GARCH Modeling")
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### GARCH(1,1) Parameter Estimates")
        garch_data = pd.DataFrame({
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'mu': [0.041, 0.090, 0.084, 0.050, 0.082, 0.075],
            'omega': [0.037, 0.057, 0.054, 0.045, 0.056, 0.125],
            'alpha': [0.051, 0.068, 0.088, 0.086, 0.076, 0.091],
            'beta': [0.925, 0.902, 0.889, 0.870, 0.902, 0.871],
            'Persistence': [0.977, 0.970, 0.977, 0.955, 0.978, 0.963]
        })
        st.dataframe(garch_data, use_container_width=True)
        st.info("**Persistence (α+β ≈ 0.97):** Volatility shocks decay slowly")
    
    with tab2:
        st.markdown("### GARCH Persistence Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        assets = ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes']
        alphas = [0.051, 0.068, 0.088, 0.086, 0.076, 0.091]
        betas = [0.925, 0.902, 0.889, 0.870, 0.902, 0.871]
        x = np.arange(len(assets))
        ax.bar(x - 0.2, alphas, 0.4, label='Alpha (ARCH)', color='steelblue')
        ax.bar(x + 0.2, betas, 0.4, label='Beta (GARCH)', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45)
        ax.set_ylabel('Parameter Value')
        ax.legend()
        ax.set_title('GARCH(1,1) Parameters by Asset')
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.code('''
from arch import arch_model

# Fit GARCH(1,1) with Normal distribution
model = arch_model(returns, mean='Constant', vol='GARCH', 
                   p=1, q=1, dist='Normal')
result = model.fit(disp='off')

print(result.summary())
print(f"Persistence: {result.params['alpha[1]'] + result.params['beta[1]']:.4f}")
        ''', language="python")


# =============================================================================
# PART 3: VaR NORMAL & STUDENT-T
# =============================================================================
elif selected_part == "part3":
    st.markdown("# ⚠️ Part 3: VaR with Normal & Student-t")
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### VaR Backtesting Results (95% Confidence)")
        var_data = pd.DataFrame({
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Normal Violations': [10, 14, 13, 20, 7, 10],
            'Normal Rate': ['3.85%', '5.38%', '5.00%', '7.69%', '2.69%', '3.85%'],
            'Student-t Violations': [2, 4, 4, 5, 0, 1],
            'Student-t Rate': ['0.77%', '1.54%', '1.54%', '1.92%', '0.00%', '0.38%']
        })
        st.dataframe(var_data, use_container_width=True)
        st.warning("**Expected Rate: 5%** - Normal is closer; Student-t is too conservative.")
    
    with tab2:
        st.markdown("### Violation Rate Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        assets = ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes']
        normal_rates = [3.85, 5.38, 5.00, 7.69, 2.69, 3.85]
        t_rates = [0.77, 1.54, 1.54, 1.92, 0.00, 0.38]
        x = np.arange(len(assets))
        ax.bar(x - 0.2, normal_rates, 0.4, label='Normal', color='steelblue')
        ax.bar(x + 0.2, t_rates, 0.4, label='Student-t', color='coral')
        ax.axhline(y=5, color='red', linestyle='--', linewidth=2, label='Expected (5%)')
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45)
        ax.set_ylabel('Violation Rate (%)')
        ax.legend()
        ax.set_title('VaR Violation Rates: Normal vs Student-t')
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.code('''
from scipy.stats import norm, t as student_t

# VaR Case 1: Normal
z_alpha = norm.ppf(0.05)
var_normal = mu + z_alpha * sigma

# VaR Case 2: Student-t
t_alpha = student_t.ppf(0.05, df=nu)
var_t = mu + t_alpha * sigma * np.sqrt((nu-2)/nu)
        ''', language="python")


# =============================================================================
# PART 4: VaR WITH GEV
# =============================================================================
elif selected_part == "part4":
    st.markdown("# 🎯 Part 4: VaR with GEV")
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### GEV Parameters")
        gev_data = pd.DataFrame({
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Extremal Index': [0.931, 0.950, 0.939, 0.962, 0.962, 0.946],
            'xi (shape)': [0.202, 0.242, 0.054, 0.130, 0.085, 0.178],
            'mu (location)': [1.466, 1.496, 1.662, 1.624, 1.643, 1.549],
            'sigma (scale)': [0.663, 0.609, 0.644, 0.620, 0.541, 0.636]
        })
        st.dataframe(gev_data, use_container_width=True)
        st.info("All ξ > 0: Heavy-tailed (Fréchet-type) distributions")
    
    with tab2:
        st.markdown("### GEV Shape Parameter by Asset")
        fig, ax = plt.subplots(figsize=(10, 6))
        assets = ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes']
        xi_values = [0.202, 0.242, 0.054, 0.130, 0.085, 0.178]
        colors = ['coral' if x > 0 else 'steelblue' for x in xi_values]
        ax.bar(assets, xi_values, color=colors, alpha=0.8, edgecolor='black')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.set_ylabel('ξ (Shape Parameter)')
        ax.set_title('GEV Shape Parameter (ξ > 0 = Heavy Tails)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.code('''
from scipy.stats import genextreme

# Extract block maxima (21-day blocks)
block_maxima = [np.max(data[i*21:(i+1)*21]) for i in range(n_blocks)]

# Fit GEV
params = genextreme.fit(block_maxima)
xi = -params[0]  # shape
mu = params[1]   # location
sigma = params[2] # scale
        ''', language="python")


# =============================================================================
# PART 5: VaR WITH GPD
# =============================================================================
elif selected_part == "part5":
    st.markdown("# 📊 Part 5: VaR with GPD")
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### GPD Parameters")
        gpd_data = pd.DataFrame({
            'Asset': ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes'],
            'Threshold': [1.066, 1.134, 1.208, 1.180, 1.209, 1.160],
            'Exceedances': [522, 522, 522, 522, 522, 522],
            'xi (shape)': [0.212, 0.208, 0.050, 0.089, 0.073, 0.164],
            'sigma (scale)': [0.546, 0.528, 0.624, 0.613, 0.549, 0.556]
        })
        st.dataframe(gpd_data, use_container_width=True)
    
    with tab2:
        st.markdown("### GPD Parameters Comparison")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        assets = ['Walmart', 'Costco', 'HomeDepot', 'Pepsico', 'TJX', 'Lowes']
        xi = [0.212, 0.208, 0.050, 0.089, 0.073, 0.164]
        sigma = [0.546, 0.528, 0.624, 0.613, 0.549, 0.556]
        
        axes[0].bar(assets, xi, color='coral', alpha=0.8, edgecolor='black')
        axes[0].set_title('GPD Shape Parameter (ξ)')
        axes[0].set_ylabel('ξ')
        plt.sca(axes[0])
        plt.xticks(rotation=45)
        
        axes[1].bar(assets, sigma, color='steelblue', alpha=0.8, edgecolor='black')
        axes[1].set_title('GPD Scale Parameter (σ)')
        axes[1].set_ylabel('σ')
        plt.sca(axes[1])
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.code('''
from scipy.stats import genpareto

# Select threshold (90th percentile)
threshold = np.quantile(-z, 0.90)

# Extract exceedances
exceedances = (-z)[(-z) > threshold] - threshold

# Fit GPD
params = genpareto.fit(exceedances, floc=0)
xi, sigma = params[0], params[2]
        ''', language="python")


# =============================================================================
# PART 6: BACKTESTING
# =============================================================================
elif selected_part == "part6":
    st.markdown("# ✅ Part 6: Backtesting Evaluation")
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        st.markdown("### Overall Backtesting Results (95% VaR)")
        results = pd.DataFrame({
            'Model': ['Normal', 'GPD', 'Student-t', 'GEV (Dep)', 'GEV (Indep)'],
            'UC Pass': ['6/6', '5/6', '0/6', '0/6', '0/6'],
            'I Pass': ['5/6', '6/6', '6/6', '6/6', '6/6'],
            'CC Pass': ['6/6', '6/6', '0/6', '0/6', '0/6'],
            'Berk Pass': ['6/6', '6/6', '6/6', '6/6', '6/6'],
            'DQ Pass': ['6/6', '6/6', '5/6', '0/6', '0/6'],
            'Score': ['96.7%', '96.7%', '56.7%', '40.0%', '40.0%']
        })
        st.dataframe(results, use_container_width=True)
        st.success("**Preferred Models:** GARCH-Normal and GARCH-GPD (tied at 96.7%)")
    
    with tab2:
        st.markdown("### Backtesting Pass Rates")
        fig, ax = plt.subplots(figsize=(10, 6))
        models = ['Normal', 'GPD', 'Student-t', 'GEV (Dep)', 'GEV (Indep)']
        scores = [96.7, 96.7, 56.7, 40.0, 40.0]
        colors = ['green' if s > 80 else 'orange' if s > 50 else 'red' for s in scores]
        ax.barh(models, scores, color=colors, alpha=0.8, edgecolor='black')
        ax.axvline(x=80, color='green', linestyle='--', label='Good (80%)')
        ax.set_xlabel('Pass Rate (%)')
        ax.set_title('Model Performance in Backtesting')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab3:
        st.code('''
# Kupiec UC Test
def kupiec_uc_test(hits, alpha):
    n, x = len(hits), hits.sum()
    p = 1 - alpha
    lr_uc = -2 * (np.log((1-p)**(n-x) * p**x) - 
                  np.log((1-x/n)**(n-x) * (x/n)**x))
    return 1 - chi2.cdf(lr_uc, 1)

# Christoffersen Independence Test
# Dynamic Quantile Test
# Model Confidence Set
        ''', language="python")


# =============================================================================
# PART 7: EXPECTED SHORTFALL
# =============================================================================
elif selected_part == "part7":
    st.markdown("# 💰 Part 7: Expected Shortfall")
    
    tab1, tab2, tab3 = st.tabs(["📊 Results", "📈 Charts", "💻 Code"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**95% Confidence**")
            es_95 = pd.DataFrame({
                'Asset': ['Lowes', 'HomeDepot', 'TJX', 'Costco', 'Walmart', 'Pepsico'],
                'VaR': ['-2.73%', '-2.42%', '-2.31%', '-2.01%', '-1.79%', '-1.62%'],
                'ES': ['-4.29%', '-3.73%', '-3.68%', '-3.19%', '-2.86%', '-2.63%']
            })
            st.dataframe(es_95, use_container_width=True)
        
        with col2:
            st.markdown("**99% Confidence**")
            es_99 = pd.DataFrame({
                'Asset': ['Lowes', 'TJX', 'HomeDepot', 'Costco', 'Walmart', 'Pepsico'],
                'VaR': ['-5.06%', '-4.15%', '-4.43%', '-3.71%', '-3.23%', '-3.00%'],
                'ES': ['-7.21%', '-6.23%', '-6.18%', '-5.62%', '-5.09%', '-4.54%']
            })
            st.dataframe(es_99, use_container_width=True)
    
    with tab2:
        st.markdown("### VaR vs ES Comparison (95%)")
        fig, ax = plt.subplots(figsize=(10, 6))
        assets = ['Lowes', 'HomeDepot', 'TJX', 'Costco', 'Walmart', 'Pepsico']
        var_vals = [-2.73, -2.42, -2.31, -2.01, -1.79, -1.62]
        es_vals = [-4.29, -3.73, -3.68, -3.19, -2.86, -2.63]
        x = np.arange(len(assets))
        ax.bar(x - 0.2, var_vals, 0.4, label='VaR', color='steelblue')
        ax.bar(x + 0.2, es_vals, 0.4, label='ES', color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(assets, rotation=45)
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.set_title('VaR vs Expected Shortfall (95%)')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Risk Ranking")
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        categories = assets
        N = len(categories)
        es_norm = [abs(v)/max(abs(v) for v in es_vals) for v in es_vals]
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        es_norm += es_norm[:1]
        ax.plot(angles, es_norm, 'o-', linewidth=2, color='coral')
        ax.fill(angles, es_norm, alpha=0.25, color='coral')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title('Risk Comparison (Higher = Riskier)')
        st.pyplot(fig)
    
    with tab3:
        st.code('''
# Historical Simulation ES
def compute_es(returns, alpha=0.05):
    var = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var]
    es = np.mean(tail)
    return var, es
        ''', language="python")


# Footer
st.sidebar.markdown("---")
st.sidebar.caption("QRM Analysis Dashboard v1.0")
