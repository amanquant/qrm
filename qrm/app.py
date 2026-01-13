import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import inspect

# --- Page Configuration ---
st.set_page_config(page_title="QRM Interactive App", layout="wide")

st.title("📊 QRM: Interactive Risk Management")
st.markdown("""
This application allows you to explore Quantitative Risk Management (QRM) models. 
Select a module from the sidebar to calculate risk metrics, view the underlying code, and analyze visual results.
""")

# --- Model 1: Parametric Value at Risk (VaR) ---
def parametric_var(portfolio_value, mean_return, std_dev, confidence_level):
    """
    Calculates the Parametric Value at Risk (VaR) for a portfolio.
    
    Parameters:
    - portfolio_value: Total value of the portfolio.
    - mean_return: Expected average return (percentage).
    - std_dev: Volatility (standard deviation) of returns.
    - confidence_level: The confidence level (e.g., 0.95 or 0.99).
    
    Returns:
    - VaR: The maximum expected loss.
    """
    alpha = 1 - confidence_level
    # Z-score for the given confidence level
    z_score = stats.norm.ppf(alpha)
    
    # VaR formula: Portfolio Value * (Mean - Z * Std_Dev)
    # Note: We focus on the loss tail, so we look at the negative return.
    # A common simplified version is Z * Volatility * Value (assuming zero mean for short horizons)
    # Here we use the full normal distribution assumption.
    cutoff_return = mean_return + z_score * std_dev
    var_amount = portfolio_value * -cutoff_return
    
    return max(var_amount, 0), cutoff_return

# --- Model 2: Monte Carlo Simulation ---
def monte_carlo_simulation(portfolio_value, mean_return, std_dev, time_horizon, simulations):
    """
    Performs a Monte Carlo Simulation to project future portfolio values.
    
    Parameters:
    - portfolio_value: Current portfolio value.
    - mean_return: Annualized expected return.
    - std_dev: Annualized volatility.
    - time_horizon: Time horizon in years.
    - simulations: Number of simulation runs.
    
    Returns:
    - final_values: Array of simulated portfolio values at the end of the horizon.
    """
    dt = time_horizon
    # Generate random shocks
    shocks = np.random.normal(0, 1, simulations)
    
    # Geometric Brownian Motion formula
    # S_t = S_0 * exp((mu - 0.5 * sigma^2)*t + sigma * sqrt(t) * Z)
    drift = (mean_return - 0.5 * std_dev ** 2) * dt
    diffusion = std_dev * np.sqrt(dt) * shocks
    
    final_values = portfolio_value * np.exp(drift + diffusion)
    return final_values

# --- Sidebar Navigation ---
st.sidebar.header("Configuration")
module = st.sidebar.selectbox("Select Model", ["Parametric VaR", "Monte Carlo Stress Test"])

# --- Main Application Logic ---

if module == "Parametric VaR":
    st.header("Parametric Value at Risk (VaR)")
    
    # 1. Inputs
    col1, col2 = st.columns(2)
    with col1:
        port_val = st.number_input("Portfolio Value ($)", value=1000000)
        conf_level = st.slider("Confidence Level", 0.90, 0.999, 0.95, 0.001)
    with col2:
        mu = st.number_input("Expected Mean Return (Daily %)", value=0.0005, format="%.4f")
        sigma = st.number_input("Volatility (Daily %)", value=0.015, format="%.3f")

    if st.button("Calculate VaR"):
        # Calculate
        var_result, cutoff_ret = parametric_var(port_val, mu, sigma, conf_level)
        
        # Display Tabs: Answer, Code, Chart
        tab_ans, tab_code, tab_chart = st.tabs(["💡 Answer", "📜 Code", "📈 Results Chart"])
        
        with tab_ans:
            st.metric(label=f"Value at Risk ({conf_level*100:.1f}%)", value=f"${var_result:,.2f}")
            st.info(f"At a {conf_level*100:.1f}% confidence level, the maximum expected loss is **${var_result:,.2f}**. This corresponds to a return cutoff of **{cutoff_ret*100:.2f}%**.")
            
        with tab_code:
            st.markdown("**Python Implementation:**")
            st.code(inspect.getsource(parametric_var), language="python")
            
        with tab_chart:
            st.markdown("**Return Distribution:**")
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            y = stats.norm.pdf(x, mu, sigma)
            ax.plot(x, y, label='Normal Distribution of Returns')
            
            # Fill the VaR area
            x_fill = np.linspace(mu - 4*sigma, cutoff_ret, 50)
            y_fill = stats.norm.pdf(x_fill, mu, sigma)
            ax.fill_between(x_fill, y_fill, color='red', alpha=0.5, label=f'VaR region ({1-conf_level:.1%} prob)')
            
            ax.axvline(cutoff_ret, color='red', linestyle='dashed')
            ax.set_title("Portfolio Return Distribution & VaR Threshold")
            ax.set_xlabel("Return")
            ax.set_ylabel("Probability Density")
            ax.legend()
            st.pyplot(fig)

elif module == "Monte Carlo Stress Test":
    st.header("Monte Carlo Stress Testing")
    
    # 1. Inputs
    col1, col2 = st.columns(2)
    with col1:
        mc_port_val = st.number_input("Initial Portfolio Value ($)", value=1000000)
        mc_sims = st.slider("Number of Simulations", 100, 10000, 1000)
    with col2:
        mc_mu = st.number_input("Annualized Drift", value=0.05)
        mc_sigma = st.number_input("Annualized Volatility", value=0.20)
        mc_time = st.number_input("Time Horizon (Years)", value=1.0)

    if st.button("Run Simulation"):
        # Calculate
        final_values = monte_carlo_simulation(mc_port_val, mc_mu, mc_sigma, mc_time, mc_sims)
        
        # Display Tabs: Answer, Code, Chart
        tab_ans, tab_code, tab_chart = st.tabs(["💡 Answer", "📜 Code", "📈 Results Chart"])
        
        with tab_ans:
            mean_final = np.mean(final_values)
            var_95 = np.percentile(final_values, 5)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Expected Value", f"${mean_final:,.2f}", delta=f"{(mean_final/mc_port_val - 1)*100:.1f}%")
            c2.metric("Worst 5% Case (VaR 95)", f"${var_95:,.2f}")
            c3.metric("Max Drawdown Sim", f"${np.min(final_values):,.2f}")
            
        with tab_code:
            st.markdown("**Python Implementation:**")
            st.code(inspect.getsource(monte_carlo_simulation), language="python")
            
        with tab_chart:
            st.markdown("**Simulation Results Distribution:**")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.hist(final_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
            ax.axvline(mean_final, color='green', linestyle='dashed', linewidth=2, label=f'Mean: ${mean_final:,.0f}')
            ax.axvline(var_95, color='red', linestyle='dashed', linewidth=2, label=f'5th Percentile: ${var_95:,.0f}')
            ax.set_title(f"Distribution of Portfolio Values after {mc_time} Year(s)")
            ax.set_xlabel("Portfolio Value ($)")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
