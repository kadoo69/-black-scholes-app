import streamlit as st
import sys
import subprocess
import numpy as np
import pandas as pd
import altair as alt

# Initialize norm, brentq, and plotly_go to None.
# They will be assigned if their respective libraries are imported successfully.
norm = None
brentq = None
plotly_go = None

st.write("--- Debugging Information ---")
st.write("Python executable:", sys.executable)
st.write("Python path:", sys.path)

st.write("--- Installed Packages (pip freeze) ---")
try:
    result = subprocess.run(['pip', 'freeze'], capture_output=True, text=True, check=True)
    st.code(result.stdout)
except Exception as e:
    st.error(f"Error running pip freeze: {e}")

# Conditional import for scipy - moved after pip freeze
try:
    import scipy
    st.write("Scipy version:", scipy.__version__)
    from scipy.stats import norm
    from scipy.optimize import brentq
except ImportError as e:
    st.error(f"Scipy import error: {e}. This app requires scipy to function. Please check your requirements.txt and Streamlit Cloud logs.")
    # Do NOT st.stop() here, we want to see the full debugging output
    norm = None # Ensure norm is None if import fails
    brentq = None # Ensure brentq is None if import fails

# Conditional import for plotly
try:
    import plotly.graph_objects as go
    plotly_go = go
    st.write("Plotly version: Successfully imported.")
except ImportError as e:
    st.warning(f"Plotly import warning: {e}. 3D plots and heatmaps will be disabled.")

st.write("--- End Debugging Information ---")


# Cumulative standard normal distribution function
def N(x):
    # Ensure norm is not None before calling it
    if norm is None:
        st.error("Error: scipy.stats.norm is not available. Cannot calculate N(x).")
        st.stop() # Stop the app here if scipy is truly missing and N is called
    return norm.cdf(x)

# Probability density function of the standard normal distribution
def phi(x):
    # Ensure norm is not None before calling its pdf method
    if norm is None:
        st.error("Error: scipy.stats.norm is not available. Cannot calculate phi(x).")
        st.stop() # Stop the app here if scipy is truly missing and phi is called
    return norm.pdf(x)

# ... rest of your black_scholes_app.py code ...


@st.cache_data
def calculate_black_scholes(S, K, T, r, sigma, q, option_type):
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Input parameters S, K, T, and sigma must be positive.")
    if r < 0 or q < 0:
        raise ValueError("Interest rate and dividend yield cannot be negative.")

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)
        delta = np.exp(-q * T) * N(d1)
        theta = -(S * np.exp(-q * T) * phi(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * N(d2) + q * S * np.exp(-q * T) * N(d1)
        rho = K * T * np.exp(-r * T) * N(d2)
    else:  # put
        price = K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)
        delta = np.exp(-q * T) * (N(d1) - 1)
        theta = -(S * np.exp(-q * T) * phi(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * N(-d2) - q * S * np.exp(-q * T) * N(-d1)
        rho = -K * T * np.exp(-r * T) * N(-d2)

    gamma = (phi(d1) * np.exp(-q * T)) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * phi(d1) * np.sqrt(T)

    intrinsic_value = max(0, S - K) if option_type == 'call' else max(0, K - S)
    time_value = max(0, price - intrinsic_value)

    if option_type == 'call':
        moneyness = 'ITM' if S > K else ('ATM' if S == K else 'OTM')
    else:  # put
        moneyness = 'ITM' if S < K else ('ATM' if S == K else 'OTM')

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "intrinsic_value": intrinsic_value,
        "time_value": time_value,
        "moneyness": moneyness,
    }

# Function to calculate Black-Scholes price for a given volatility (used for IV)
@st.cache_data
def bs_price_for_iv(sigma, S, K, T, r, q, option_type):
    try:
        # Ensure sigma is positive for calculation
        if sigma <= 0: return 1e10 # Return a large number to push solver away from non-positive sigma
        return calculate_black_scholes(S, K, T, r, sigma, q, option_type)['price']
    except ValueError:
        return 1e10 # Handle cases where sigma might lead to invalid inputs

@st.cache_data
def find_implied_volatility(market_price, S, K, T, r, q, option_type):
    # Define the function whose root we want to find
    # This function returns the difference between the Black-Scholes price and the market price
    def f(sigma):
        return bs_price_for_iv(sigma, S, K, T, r, q, option_type) - market_price

    # Find a bracket [a, b] such that f(a) and f(b) have opposite signs
    # This is crucial for brentq to work
    a = 0.001 # Minimum reasonable volatility
    b = 5.0   # Maximum reasonable volatility (500%)

    # Check if a root exists within the bracket
    if f(a) * f(b) >= 0:
        # Try to expand the bracket or return an error if no root is found
        # For simplicity, we'll try a wider range or indicate failure
        # In a real app, you might want more sophisticated bracketing
        try:
            # Attempt to find a bracket by iterating
            vol_test = np.linspace(0.001, 5.0, 100)
            f_vals = [f(v) for v in vol_test]
            
            # Find first sign change
            idx = np.where(np.diff(np.sign(f_vals)))[0]
            if len(idx) > 0:
                a = vol_test[idx[0]]
                b = vol_test[idx[0] + 1]
            else:
                raise ValueError("Could not find a suitable bracket for implied volatility.")
        except Exception:
            raise ValueError("Could not find a suitable bracket for implied volatility.")

    # Use brentq to find the root (implied volatility)
    implied_vol = brentq(f, a, b)
    return implied_vol

# Function to calculate P/L at expiration
@st.cache_data
def calculate_pnl_at_expiration(spot_prices, K, option_type, premium):
    pnl_data = []
    for S_at_exp in spot_prices:
        if option_type == 'call':
            pnl = max(0, S_at_exp - K) - premium
        else: # put
            pnl = max(0, K - S_at_exp) - premium
        pnl_data.append({'Stock Price at Expiration': S_at_exp, 'Profit/Loss': pnl})
    return pd.DataFrame(pnl_data)

st.set_page_config(layout="wide", page_title="Black-Scholes Options Calculator")

st.title("Black-Scholes Options Pricing Model")

st.sidebar.header("Input Parameters")

with st.sidebar.expander("Current Stock Price (S)", expanded=True):
    S = st.slider("", min_value=10.0, max_value=200.0, value=100.0, step=1.0, key='S_slider')
    st.write("The current market price of the underlying asset.")

with st.sidebar.expander("Strike Price (K)", expanded=True):
    K = st.slider("", min_value=10.0, max_value=200.0, value=100.0, step=1.0, key='K_slider')
    st.write("The price at which the option holder can buy (call) or sell (put) the underlying asset.")

with st.sidebar.expander("Time to Expiration (T, years)", expanded=True):
    T = st.slider("", min_value=0.01, max_value=5.0, value=1.0, step=0.01, key='T_slider')
    st.write("The remaining time until the option expires, expressed in years.")

with st.sidebar.expander("Risk-free Interest Rate (r, %)", expanded=True):
    r = st.slider("", min_value=0.0, max_value=10.0, value=5.0, step=0.1, key='r_slider') / 100
    st.write("The theoretical rate of return of an investment with zero risk, expressed as an annualized percentage.")

with st.sidebar.expander("Volatility (σ, %)", expanded=True):
    sigma = st.slider("", min_value=0.01, max_value=100.0, value=20.0, step=0.1, key='sigma_slider') / 100
    st.write("A measure of the expected fluctuation in the underlying asset's price, expressed as an annualized percentage.")

with st.sidebar.expander("Dividend Yield (q, %)", expanded=True):
    q = st.slider("", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key='q_slider') / 100
    st.write("The annualized dividend yield of the underlying asset. Defaults to 0 if not specified.")

with st.sidebar.expander("Option Type", expanded=True):
    option_type = st.radio("", ('call', 'put'), key='option_type_radio')
    st.write("Select whether you are pricing a Call option (right to buy) or a Put option (right to sell).")

with st.sidebar.expander("Market Option Price (for Implied Volatility)", expanded=True):
    market_price = st.number_input("", min_value=0.0, format="%.2f", key='market_price_input')
    st.write("Enter the current market price of the option to calculate its Implied Volatility.")


try:
    results = calculate_black_scholes(S, K, T, r, sigma, q, option_type)

    with st.expander("Calculated Results", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Option Price", f"{results['price']:.4f}")
            with st.popover("What is this?"):
                st.markdown("The theoretical fair value of the option according to the Black-Scholes model.")
        with col2:
            st.metric("Intrinsic Value", f"{results['intrinsic_value']:.4f}")
            with st.popover("What is this?"):
                st.markdown("The immediate profit if the option were exercised today (e.g., for a call, max(0, S - K)).")
        with col3:
            st.metric("Time Value", f"{results['time_value']:.4f}")
            with st.popover("What is this?"):
                st.markdown("The portion of the option's price attributed to the remaining time until expiration and volatility.")
        st.write(f"**Moneyness:** {results['moneyness']}")
        with st.popover("What is this?"):
            st.markdown("Indicates if the option is In-The-Money (ITM), At-The-Money (ATM), or Out-of-The-Money (OTM).")

    with st.expander("Greeks", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Delta", f"{results['delta']:.4f}")
            with st.popover("What is this?"):
                st.markdown("Measures the sensitivity of the option price to a $1 change in the underlying stock price.")
        with col2:
            st.metric("Gamma", f"{results['gamma']:.4f}")
            with st.popover("What is this?"):
                st.markdown("Measures the rate of change of Delta with respect to a change in the underlying stock price.")
        with col3:
            st.metric("Theta", f"{results['theta']:.4f}")
            with st.popover("What is this?"):
                st.markdown("Measures the sensitivity of the option price to the passage of time (time decay).")
        with col4:
            st.metric("Vega", f"{results['vega']:.4f}")
            with st.popover("What is this?"):
                st.markdown("Measures the sensitivity of the option price to a 1% change in the underlying asset's volatility.")
        with col5:
            st.metric("Rho", f"{results['rho']:.4f}")
            with st.popover("What is this?"):
                st.markdown("Measures the sensitivity of the option price to a 1% change in the risk-free interest rate.")

    # Implied Volatility Calculation and Display
    if market_price > 0:
        try:
            implied_vol = find_implied_volatility(market_price, S, K, T, r, q, option_type)
            with st.expander("Implied Volatility", expanded=True):
                st.metric("Implied Volatility (IV)", f"{implied_vol * 100:.2f}%")
                with st.popover("What is this?"):
                    st.markdown("The volatility implied by the current market price of the option, given the other Black-Scholes inputs.")
        except ValueError as e:
            st.warning(f"Could not calculate Implied Volatility: {e}")
        except Exception as e:
            st.warning(f"Error calculating Implied Volatility: {e}")

    st.subheader("Sensitivity Analysis")
    
    # Option Price and Delta vs. Underlying Stock Price Chart
    stock_prices_range = np.linspace(S * 0.5, S * 1.5, 100) # Range around current stock price
    chart_data_stock_price = []
    for price in stock_prices_range:
        try:
            calc_res = calculate_black_scholes(price, K, T, r, sigma, q, option_type)
            chart_data_stock_price.append({
                'Stock Price': price,
                'Option Price': calc_res['price'],
                'Delta': calc_res['delta'],
                'Gamma': calc_res['gamma']
            })
        except ValueError:
            continue
    df_stock_price_charts = pd.DataFrame(chart_data_stock_price)

    chart_option_price = alt.Chart(df_stock_price_charts).mark_line().encode(
        x=alt.X('Stock Price', axis=alt.Axis(title='Underlying Stock Price')),
        y=alt.Y('Option Price', axis=alt.Axis(title='Option Price')),
        tooltip=['Stock Price', 'Option Price']
    ).properties(
        title='Option Price vs. Underlying Stock Price'
    ).interactive()
    st.altair_chart(chart_option_price, use_container_width=True)

    chart_delta = alt.Chart(df_stock_price_charts).mark_line(color='red').encode(
        x=alt.X('Stock Price', axis=alt.Axis(title='Underlying Stock Price')),
        y=alt.Y('Delta', axis=alt.Axis(title='Delta')),
        tooltip=['Stock Price', 'Delta']
    ).properties(
        title='Delta vs. Underlying Stock Price'
    ).interactive()
    st.altair_chart(chart_delta, use_container_width=True)

    # Gamma vs. Underlying Stock Price Chart
    chart_gamma = alt.Chart(df_stock_price_charts).mark_line(color='green').encode(
        x=alt.X('Stock Price', axis=alt.Axis(title='Underlying Stock Price')),
        y=alt.Y('Gamma', axis=alt.Axis(title='Gamma')),
        tooltip=['Stock Price', 'Gamma']
    ).properties(
        title='Gamma vs. Underlying Stock Price'
    ).interactive()
    st.altair_chart(chart_gamma, use_container_width=True)

    # Theta vs. Time to Expiration Chart
    time_to_exp_range = np.linspace(0.01, T * 2, 100) # Range around current Time to Expiration
    chart_data_time = []
    for time_val in time_to_exp_range:
        try:
            calc_res = calculate_black_scholes(S, K, time_val, r, sigma, q, option_type)
            chart_data_time.append({
                'Time to Expiration': time_val,
                'Theta': calc_res['theta']
            })
        except ValueError:
            continue
    df_time_charts = pd.DataFrame(chart_data_time)

    chart_theta = alt.Chart(df_time_charts).mark_line(color='purple').encode(
        x=alt.X('Time to Expiration', axis=alt.Axis(title='Time to Expiration (Years)')),
        y=alt.Y('Theta', axis=alt.Axis(title='Theta')),
        tooltip=['Time to Expiration', 'Theta']
    ).properties(
        title='Theta vs. Time to Expiration'
    ).interactive()
    st.altair_chart(chart_theta, use_container_width=True)

    # Vega vs. Volatility Chart
    volatility_range = np.linspace(0.01, sigma * 2, 100) # Range around current Volatility
    chart_data_vol = []
    for vol_val in volatility_range:
        try:
            calc_res = calculate_black_scholes(S, K, T, r, vol_val, q, option_type)
            chart_data_vol.append({
                'Volatility': vol_val,
                'Vega': calc_res['vega']
            })
        except ValueError:
            continue
    df_vol_charts = pd.DataFrame(chart_data_vol)

    chart_vega = alt.Chart(df_vol_charts).mark_line(color='orange').encode(
        x=alt.X('Volatility', axis=alt.Axis(title='Volatility (%)')),
        y=alt.Y('Vega', axis=alt.Axis(title='Vega')),
        tooltip=['Volatility', 'Vega']
    ).properties(
        title='Vega vs. Volatility'
    ).interactive()
    st.altair_chart(chart_vega, use_container_width=True)

    # 3D Surface Plot: Option Price vs. Time vs. Volatility
    st.subheader("3D Option Price Surface")
    
    if plotly_go: # Check if plotly was successfully imported
        # Define ranges for the 3D plot
        time_range_3d = np.linspace(0.01, 2.0, 30) # Time from 0.01 to 2 years
        vol_range_3d = np.linspace(0.05, 1.0, 30) # Volatility from 5% to 100%

        Z = np.zeros((len(time_range_3d), len(vol_range_3d)))

        for i, t_val in enumerate(time_range_3d):
            for j, vol_val in enumerate(vol_range_3d):
                try:
                    # Use the current S, K, r, q, option_type from sidebar
                    price_3d = calculate_black_scholes(S, K, t_val, r, vol_val, q, option_type)['price']
                    Z[i, j] = price_3d
                except ValueError:
                    Z[i, j] = np.nan # Handle invalid inputs in the loop

        fig = plotly_go.Figure(data=[plotly_go.Surface(z=Z, x=vol_range_3d * 100, y=time_range_3d)])
        fig.update_layout(
            title='Option Price Surface (Volatility vs. Time)',
            scene = dict(
                xaxis_title='Volatility (%)',
                yaxis_title='Time to Expiration (Years)',
                zaxis_title='Option Price',
                aspectratio=dict(x=1, y=1, z=0.7),
                camera_eye=dict(x=1.8, y=1.8, z=0.8)
            )
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("3D Option Price Surface not available (Plotly not imported).")

    # Heatmap: Option Price vs. Underlying Stock Price vs. Volatility
    st.subheader("Option Price Heatmap (Stock Price vs. Volatility)")

    if plotly_go: # Check if plotly was successfully imported
        # Define ranges for the heatmap
        heatmap_S_range = np.linspace(S * 0.7, S * 1.3, 50) # Stock Price range
        heatmap_sigma_range = np.linspace(0.05, 0.8, 50) # Volatility range (5% to 80%)

        Z_heatmap = np.zeros((len(heatmap_sigma_range), len(heatmap_S_range)))

        for i, current_sigma in enumerate(heatmap_sigma_range):
            for j, current_S in enumerate(heatmap_S_range):
                try:
                    # Calculate option price for each combination
                    price_hm = calculate_black_scholes(current_S, K, T, r, current_sigma, q, option_type)['price']
                    Z_heatmap[i, j] = price_hm
                except ValueError:
                    Z_heatmap[i, j] = np.nan # Handle invalid inputs

        fig_heatmap = plotly_go.Figure(data=plotly_go.Heatmap(
            z=Z_heatmap,
            x=heatmap_S_range,
            y=heatmap_sigma_range * 100, # Display volatility as percentage
            colorscale='RdYlGn', # Red-Yellow-Green for comparison
            colorbar=dict(title='Option Price')
        ))

        fig_heatmap.update_layout(
            title='Option Price Heatmap',
            xaxis_title='Underlying Stock Price',
            yaxis_title='Volatility (%)'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Option Price Heatmap not available (Plotly not imported).")

    # P/L Diagram
    st.subheader("Profit/Loss Diagram at Expiration")
    pnl_spot_prices = np.linspace(S * 0.5, S * 1.5, 100)
    df_pnl = calculate_pnl_at_expiration(pnl_spot_prices, K, option_type, results['price'])

    pnl_chart = alt.Chart(df_pnl).mark_line().encode(
        x=alt.X('Stock Price at Expiration', axis=alt.Axis(title='Stock Price at Expiration')),
        y=alt.Y('Profit/Loss', axis=alt.Axis(title='Profit/Loss')),
        tooltip=['Stock Price at Expiration', 'Profit/Loss']
    ).properties(
        title='Profit/Loss at Expiration'
    ).interactive()

    # Add a horizontal line at P/L = 0 for breakeven visualization
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='gray', strokeDash=[3, 3]).encode(y='y')

    st.altair_chart(pnl_chart + zero_line, use_container_width=True)

except ValueError as e:
    st.error(f"Input Error: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

st.markdown("---")
st.markdown("Disclaimer: This calculator is for educational purposes only and should not be used for actual trading decisions.")
