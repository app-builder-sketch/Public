import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform

# -------------------------------------------------------------------------
# 1. HELPER FUNCTIONS & MATH IMPLEMENTATION
# -------------------------------------------------------------------------

def calculate_relativistic_ma(price_series, base_period=20, sensitivity=1.0):
    """
    Concept P2: Relativistic Time-Dilation Moving Average.
    Uses Special Relativity (Lorentz transformations) to adjust lookback.
    
    Logic:
    - Velocity (v) = Normalized momentum of price.
    - c (Speed of Light) = Theoretical max velocity (approximated by statistical bounds).
    - Lorentz Factor (gamma) = 1 / sqrt(1 - v^2/c^2)
    - Effective Period = base_period / gamma 
      (High velocity -> High Gamma -> Short Period -> Fast reaction/No Lag)
    """
    prices = price_series.values
    n = len(prices)
    
    # Calculate Velocity 'v' (Rate of Change)
    roc = np.diff(prices, prepend=prices[0])
    
    # Define 'c' (Speed of Light proxy) as a rolling max deviation (e.g., 3-sigma equivalent)
    # We use a rolling window to define the local "speed limit" of the market
    rolling_std = pd.Series(roc).rolling(window=base_period*2).std().bfill().values
    c = rolling_std * 4  # 4-sigma move is considered "speed of light" (max limit)
    
    # Calculate Beta (v/c)
    # Clip beta to 0.99 to avoid division by zero (singularity)
    beta = np.abs(roc) / (c + 1e-9) 
    beta = np.clip(beta, 0, 0.99)
    
    # Lorentz Factor (Gamma)
    gamma = 1 / np.sqrt(1 - beta**2)
    
    # Apply Time Dilation to Alpha
    # Standard EMA Alpha = 2 / (N + 1)
    # Relativistic: N becomes N / (gamma * sensitivity)
    # As gamma increases (high speed), N decreases, Alpha increases (faster tracking)
    
    rma = np.zeros(n)
    rma[0] = prices[0]
    
    for i in range(1, n):
        # Dynamic N based on relativistic velocity
        effective_n = base_period / (gamma[i] * sensitivity)
        # Clamp N to sensible bounds (e.g., minimum 1 period)
        effective_n = max(1.0, effective_n)
        
        alpha = 2 / (effective_n + 1)
        rma[i] = alpha * prices[i] + (1 - alpha) * rma[i-1]
        
    return rma, gamma

def estimate_lyapunov_exponent(price_series, window=50, embedding_dim=3, delay=1):
    """
    Concept C5: Lyapunov Exponent Stability Map.
    Estimates the Largest Lyapunov Exponent (LLE) using a simplified Rosenstein method 
    adapted for vectorized rolling windows.
    
    Logic:
    - Reconstruct phase space using time-delay embedding.
    - Measure divergence of nearest neighbors over time.
    - Positive Lambda = Chaos/Divergence.
    - Negative/Low Lambda = Stability/Convergence.
    """
    prices = np.log(price_series.values) # Work with log-prices for scaling
    n = len(prices)
    lyapunov = np.zeros(n)
    
    # Pre-compute phase space trajectory matrix
    # Vectors X(t) = [x(t), x(t-tau), x(t-2tau)...]
    # We do this efficiently by iterating through the window
    
    for t in range(window + (embedding_dim * delay), n):
        # Extract the current window
        segment = prices[t-window:t]
        
        # Simple estimation of divergence:
        # 1. Calculate derivatives (velocity in phase space)
        d1 = np.diff(segment)
        
        # 2. If the variance of derivatives is increasing, trajectories are separating
        # We model LLE as the slope of log-divergence
        # This is a heuristic proxy for the full Rosenstein algorithm for computational speed
        
        # Divide window into two halves
        half = window // 2
        d_early = np.std(d1[:half])
        d_late = np.std(d1[half:])
        
        if d_early == 0: d_early = 1e-9
        
        # Logarithmic rate of expansion
        lambda_est = np.log(d_late / d_early)
        
        lyapunov[t] = lambda_est
        
    return lyapunov

# -------------------------------------------------------------------------
# 2. MOCK DATA GENERATION
# -------------------------------------------------------------------------
np.random.seed(42)
steps = 300
t = np.linspace(0, 1, steps)
# Geometric Brownian Motion with a "Parabolic" event
price_path = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, steps)))
# Inject a parabolic surge (Relativistic event)
price_path[200:230] = price_path[200:230] * np.linspace(1.0, 1.2, 30) 
# Inject a crash
price_path[230:250] = price_path[230:250] * np.linspace(1.2, 0.9, 20)

df = pd.DataFrame({'Close': price_path})

# -------------------------------------------------------------------------
# 3. INDICATOR CALCULATION
# -------------------------------------------------------------------------

# P2: Relativistic MA
df['RMA'], df['Gamma'] = calculate_relativistic_ma(df['Close'], base_period=20)

# C5: Lyapunov Exponent
df['Lyapunov'] = estimate_lyapunov_exponent(df['Close'], window=40)

# Signal Logic
# Buy: Price > RMA AND Lyapunov < Threshold (Market is stable enough to trend)
# Sell: Price < RMA
chaos_threshold = 0.5
df['Signal'] = 0
df.loc[(df['Close'] > df['RMA']) & (df['Lyapunov'] < chaos_threshold), 'Signal'] = 1
df.loc[(df['Close'] < df['RMA']), 'Signal'] = -1

# -------------------------------------------------------------------------
# 4. VISUALIZATION (PLOTLY)
# -------------------------------------------------------------------------

fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, 
                    row_heights=[0.5, 0.25, 0.25],
                    subplot_titles=("Price Action & Relativistic MA", "Lorentz Factor (Time Dilation)", "Lyapunov Exponent (Stability)"))

# Plot Price
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='gray', width=1)), row=1, col=1)

# Plot Relativistic MA (Color coded by trend)
fig.add_trace(go.Scatter(x=df.index, y=df['RMA'], mode='lines', name='Relativistic MA (P2)', line=dict(color='cyan', width=2)), row=1, col=1)

# Plot Signals
buys = df[df['Signal'] == 1]
sells = df[df['Signal'] == -1]
fig.add_trace(go.Scatter(x=buys.index, y=buys['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=8)), row=1, col=1)
fig.add_trace(go.Scatter(x=sells.index, y=sells['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=8)), row=1, col=1)

# Plot Lorentz Factor
fig.add_trace(go.Scatter(x=df.index, y=df['Gamma'], mode='lines', name='Gamma (Time Dilation)', line=dict(color='orange')), row=2, col=1)
fig.add_hline(y=1.0, line_dash="dot", row=2, col=1, annotation_text="Newtonian Time (Rest)")

# Plot Lyapunov
fig.add_trace(go.Scatter(x=df.index, y=df['Lyapunov'], mode='lines', name='Lyapunov Exp (C5)', line=dict(color='magenta')), row=3, col=1)
fig.add_hline(y=chaos_threshold, line_dash="dot", row=3, col=1, annotation_text="Chaos Threshold", line_color="red")
fig.add_hline(y=0, line_dash="dot", row=3, col=1, annotation_text="Stable/Unstable Boundary", line_color="green")

fig.update_layout(title='Relativistic Trend Strategy with Lyapunov Chaos Filter',
                  template='plotly_dark', height=1000)

fig.show()
