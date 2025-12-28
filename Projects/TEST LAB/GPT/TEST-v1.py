import numpy as np
import pandas as pd
from scipy.special import expit  # stable logistic
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def _safe_log(x, eps=1e-12):
    return np.log(np.clip(x, eps, None))

def _rolling_cumsum_window_sum(x: np.ndarray, w: int) -> np.ndarray:
    """
    Fixed-window rolling sum via cumsum (vectorized).
    Returns length n with NaN for first w-1 entries.
    """
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)
    if w <= 0 or n == 0:
        return out
    c = np.cumsum(np.where(np.isnan(x), 0.0, x).astype(float))
    # sum[i] = c[i] - c[i-w] (with edge handling)
    out[w-1:] = c[w-1:] - np.concatenate(([0.0], c[:-w]))
    return out

def relativistic_time_dilation_ma(close: pd.Series,
                                 base_lookback: int = 50,
                                 v_lookback: int = 30,
                                 c_quantile: float = 0.995,
                                 min_lookback: int = 2) -> pd.Series:
    """
    P2: Relativistic Time-Dilation Moving Average.
    - Velocity proxy v_t = |log(C_t/C_{t-1})|
    - Market 'speed of light' proxy c = rolling-quantile of v (or global quantile for stability)
    - Lorentz factor gamma = 1/sqrt(1 - (v/c)^2)
    - Proper-time window contraction: N_eff = floor(base_lookback / gamma), clamped
      (so at high velocity, proper time advances slower => fewer bars in proper-time window => more responsive MA)

    Vectorized with variable-window sums via cumsum + per-index start offsets.
    """
    x = close.to_numpy(dtype=float)
    n = x.shape[0]
    out = np.full(n, np.nan, dtype=float)

    # velocity proxy
    r = np.diff(_safe_log(x), prepend=np.nan)
    v = np.abs(r)

    # c proxy (avoid per-bar rolling quantile cost): use a stable rolling max-like quantile proxy
    # Compute a rolling high-quantile approximation via rolling window sort on numpy would be costly.
    # Instead: use an EW "ceiling" proxy that tracks high quantiles stably.
    # This remains a market-proxy constant (as required) without assuming a physical constant.
    v_f = pd.Series(v).fillna(0.0).to_numpy()
    # robust ceiling: max over trailing v_lookback (vectorized using rolling max via pandas)
    c_proxy = pd.Series(v_f).rolling(v_lookback, min_periods=1).max().to_numpy()

    # ensure strictly positive c
    c_proxy = np.maximum(c_proxy, np.nanmedian(v_f[v_f > 0]) if np.any(v_f > 0) else 1e-6)

    beta = np.clip(v_f / c_proxy, 0.0, 0.999999)  # avoid superluminal
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    n_eff = np.floor(base_lookback / gamma).astype(int)
    n_eff = np.clip(n_eff, min_lookback, base_lookback)

    # variable-window MA using cumulative sum:
    # ma[i] = (cs[i] - cs[i - n_eff[i]]) / n_eff[i]
    cs = np.cumsum(np.where(np.isnan(x), 0.0, x))
    idx = np.arange(n)
    start = idx - n_eff
    start_clipped = np.clip(start, -1, n-1)

    # cs_start: cs[start-1] effectively; for start <= 0 use 0
    cs_start = np.where(start_clipped >= 0, cs[start_clipped], 0.0)
    sums = cs - cs_start
    denom = n_eff.astype(float)

    # mask first values where window not fully available (strict causality)
    valid = start >= 0
    out[valid] = sums[valid] / denom[valid]

    return pd.Series(out, index=close.index, name="Relativistic_TD_MA")

def thermodynamic_entropy_exhaustion(close: pd.Series,
                                    window: int = 64,
                                    nbins: int = 12,
                                    scale_window: int = 256) -> pd.Series:
    """
    P4: Thermodynamic Entropy Exhaustion via Shannon entropy of discretized return states.
    Steps:
    - returns r_t = log(C_t/C_{t-1})
    - discretize r_t into nbins using symmetric bin edges based on a rolling robust scale
    - compute rolling bin counts vectorized using one-hot cumulative sums
    - entropy H_t = -sum_k p_k log(p_k)

    Fully vectorized: O(n * nbins) with cumulative sums.
    """
    x = close.to_numpy(dtype=float)
    n = x.shape[0]

    r = np.diff(_safe_log(x), prepend=np.nan)
    r0 = np.where(np.isnan(r), 0.0, r)

    # robust scale proxy (market-proxy "temperature" scale)
    # rolling MAD-like proxy using rolling median absolute deviation via pandas (acceptable; returns vector)
    rs = pd.Series(r0)
    med = rs.rolling(scale_window, min_periods=1).median()
    mad = (rs - med).abs().rolling(scale_window, min_periods=1).median()
    sigma = (mad * 1.4826).to_numpy()
    sigma = np.maximum(sigma, np.nanmedian(sigma[sigma > 0]) if np.any(sigma > 0) else 1e-6)

    # symmetric bin edges around 0, scaled by sigma (time-varying edges)
    # For vectorization, discretize using a single global edge set based on median sigma:
    sig_ref = np.nanmedian(sigma) if np.isfinite(np.nanmedian(sigma)) else 1e-6
    L = 4.0 * sig_ref
    edges = np.linspace(-L, L, nbins + 1)

    # bin indices 0..nbins-1
    b = np.digitize(r0, edges) - 1
    b = np.clip(b, 0, nbins - 1)

    # one-hot (n x nbins) in float for cumsum
    onehot = np.zeros((n, nbins), dtype=float)
    onehot[np.arange(n), b] = 1.0

    # rolling counts via cumulative sums along time
    cs = np.cumsum(onehot, axis=0)
    counts = cs.copy()
    if window < n:
        counts[window:] = cs[window:] - cs[:-window]
    # first window-1 are partial -> NaN for strictness
    counts[:window-1, :] = np.nan

    p = counts / float(window)
    # avoid log(0)
    p_safe = np.where(np.isnan(p), np.nan, np.clip(p, 1e-12, 1.0))
    H = -np.nansum(p_safe * np.log(p_safe), axis=1)

    return pd.Series(H, index=close.index, name="Entropy_Level")

def quantum_tunneling_probability(df: pd.DataFrame,
                                 sr_window: int = 80,
                                 width_bars: int = 10,
                                 kappa_scale: float = 1.0) -> pd.Series:
    """
    P1: Quantum Tunneling Probability Oscillator.
    Market mapping:
    - Potential barrier height V0: distance from price to nearest S/R level (recent swing high/low)
    - Particle energy E: kinetic ~ 0.5 * m * v^2
      where v = |log-return| and m is volume-proxy mass (normalized volume)
    - Barrier width a: width_bars as a proxy for barrier thickness (time/structure thickness)
    - ħ proxy: median(|return|) over sr_window (market microstructure scale)
    - T ≈ exp( -2 * kappa * a ), kappa = sqrt(2m(V0 - E)) / ħ for V0 > E else ~0 => T~1

    Returns T in [0,1] (higher => more likely 'breakout'/crossing).
    """
    close = df["Close"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float) if "High" in df.columns else close
    low = df["Low"].to_numpy(dtype=float) if "Low" in df.columns else close
    vol = df["Volume"].to_numpy(dtype=float) if "Volume" in df.columns else np.ones_like(close)

    n = close.shape[0]
    T = np.full(n, np.nan, dtype=float)

    r = np.diff(_safe_log(close), prepend=np.nan)
    v = np.abs(np.where(np.isnan(r), 0.0, r))

    # mass proxy from volume (normalize)
    vmed = np.nanmedian(vol[vol > 0]) if np.any(vol > 0) else 1.0
    m = np.clip(vol / vmed, 0.1, 10.0)

    # recent S/R proxy: rolling max(high) and rolling min(low)
    sr_hi = pd.Series(high).rolling(sr_window, min_periods=sr_window).max().to_numpy()
    sr_lo = pd.Series(low).rolling(sr_window, min_periods=sr_window).min().to_numpy()

    # distance to nearest barrier (choose whichever is nearer in price-space)
    d_hi = np.abs(sr_hi - close)
    d_lo = np.abs(close - sr_lo)
    V0 = np.minimum(d_hi, d_lo)

    # energy proxy
    E = 0.5 * m * (v**2)

    # ħ proxy
    hbar = pd.Series(v).rolling(sr_window, min_periods=sr_window).median().to_numpy()
    hbar = np.maximum(hbar, np.nanmedian(v[v > 0]) if np.any(v > 0) else 1e-6)

    # barrier excess
    delta = V0 - E
    delta = np.where(np.isnan(delta), np.nan, np.maximum(delta, 0.0))

    kappa = np.sqrt(2.0 * m * delta) / hbar
    kappa = kappa_scale * kappa

    a = float(max(1, width_bars))
    expo = -2.0 * kappa * a

    # if delta==0 => kappa==0 => T=1
    T_val = np.exp(np.clip(expo, -700, 0))  # stable exp
    # strict: only valid where sr defined
    valid = np.isfinite(sr_hi) & np.isfinite(sr_lo)
    T[valid] = T_val[valid]

    return pd.Series(T, index=df.index, name="Tunneling_Prob")

def recursive_bayesian_trend_filter(features: pd.DataFrame,
                                    prior_bull: float = 0.5,
                                    lr: float = 1.0) -> pd.Series:
    """
    O5: Recursive Bayesian Trend Filter (online posterior).
    We maintain log-odds and update with feature-derived log-likelihood increments.

    Feature mapping (no standard TA):
    - z_ret: sign(returns) contribution
    - z_ma_slope: slope of relativistic MA (proper-time trend)
    - z_dH: negative entropy gradient supports trend continuation; positive supports uncertainty
    - z_T: tunneling probability supports directional continuation through barrier

    Update:
      logit(p_t) = logit(p_{t-1}) + lr * (w · z_t)
    with z normalized to stable ranges.

    Returns p_bull in [0,1].
    """
    # Expected columns: ret, ma, entropy, tunneling
    ret = features["ret"].to_numpy(dtype=float)
    ma = features["rma"].to_numpy(dtype=float)
    H = features["H"].to_numpy(dtype=float)
    T = features["T"].to_numpy(dtype=float)

    n = len(features)
    p = np.full(n, np.nan, dtype=float)

    # z components
    z_ret = np.tanh(10.0 * ret)  # map small returns into smooth [-1,1]
    ma_slope = np.diff(ma, prepend=np.nan)
    z_ma = np.tanh(5.0 * np.where(np.isnan(ma_slope), 0.0, ma_slope))

    dH = np.diff(H, prepend=np.nan)
    z_dH = -np.tanh(2.0 * np.where(np.isnan(dH), 0.0, dH))  # entropy rising => negative evidence

    z_T = 2.0 * (np.clip(np.where(np.isnan(T), 0.5, T), 0.0, 1.0) - 0.5)  # [-1,1]

    # weights (kept explicit and minimal; still within O5 Bayesian updating logic)
    w_ret, w_ma, w_dH, w_T = 0.9, 1.1, 0.7, 0.8

    inc = w_ret * z_ret + w_ma * z_ma + w_dH * z_dH + w_T * z_T

    # recursive log-odds
    logit = np.log(prior_bull / (1.0 - prior_bull))
    for i in range(n):
        if np.isfinite(inc[i]):
            logit = logit + lr * inc[i]
        p[i] = expit(logit)

    return pd.Series(p, index=features.index, name="Bayes_Bull_Prob")

def compute_quant_physics_stack(df: pd.DataFrame,
                               base_lookback: int = 50,
                               v_lookback: int = 30,
                               entropy_window: int = 64,
                               entropy_bins: int = 12,
                               sr_window: int = 80,
                               barrier_width_bars: int = 10) -> pd.DataFrame:
    """
    Convenience wrapper to compute P2, P4, P1, O5 outputs in one dataframe.
    """
    close = df["Close"]

    rma = relativistic_time_dilation_ma(
        close=close,
        base_lookback=base_lookback,
        v_lookback=v_lookback
    )

    H = thermodynamic_entropy_exhaustion(
        close=close,
        window=entropy_window,
        nbins=entropy_bins
    )

    T = quantum_tunneling_probability(
        df=df,
        sr_window=sr_window,
        width_bars=barrier_width_bars
    )

    ret = close.pct_change().fillna(0.0)  # percent return as input scale (not a TA indicator)
    feats = pd.DataFrame({"ret": ret.to_numpy(), "rma": rma.to_numpy(), "H": H.to_numpy(), "T": T.to_numpy()}, index=df.index)

    p_bull = recursive_bayesian_trend_filter(feats)

    out = df.copy()
    out["Relativistic_TD_MA"] = rma
    out["Entropy_Level"] = H
    out["Tunneling_Prob"] = T
    out["Bayes_Bull_Prob"] = p_bull
    out["Bayes_Bear_Prob"] = 1.0 - p_bull

    # helpful differentials (still derived from concepts above)
    out["Entropy_Gradient"] = out["Entropy_Level"].diff()
    out["RMA_Slope"] = out["Relativistic_TD_MA"].diff()

    return out
def generate_quant_physics_signals(df_feat: pd.DataFrame,
                                  p_enter: float = 0.65,
                                  p_exit: float = 0.45,
                                  tunneling_enter: float = 0.70,
                                  entropy_peak_lookback: int = 20,
                                  require_entropy_fall: bool = True,
                                  require_rma_slope_align: bool = True) -> pd.DataFrame:
    """
    Signals are strictly tied to P2/P4/P1/O5 triggers.

    LONG entry:
      - Bayes_Bull_Prob >= p_enter
      - Tunneling_Prob >= tunneling_enter (breakout/crossing likelihood)
      - Optional: Entropy is falling after a local peak (exhaustion resolving into structure)
      - Optional: RMA_Slope > 0 (proper-time trend alignment)

    LONG exit:
      - Bayes_Bull_Prob <= p_exit OR (Entropy rising sharply + tunneling collapses)

    SHORT entry/exit symmetric using Bayes_Bear_Prob and RMA_Slope < 0.

    Returns df with boolean columns: long_entry, long_exit, short_entry, short_exit and position.
    """
    d = df_feat.copy()

    p_bull = d["Bayes_Bull_Prob"]
    p_bear = d["Bayes_Bear_Prob"]
    T = d["Tunneling_Prob"]
    dH = d["Entropy_Gradient"]
    slope = d["RMA_Slope"]

    # entropy peak detection (local max within lookback)
    H = d["Entropy_Level"]
    H_roll_max = H.rolling(entropy_peak_lookback, min_periods=entropy_peak_lookback).max()
    recently_peaked = (H_roll_max == H_roll_max.shift(1)) | (H >= H_roll_max * 0.999)  # tolerant
    entropy_falling = (dH < 0)

    long_entry = (p_bull >= p_enter) & (T >= tunneling_enter)
    short_entry = (p_bear >= p_enter) & (T >= tunneling_enter)

    if require_entropy_fall:
        long_entry &= recently_peaked.shift(1).fillna(False) & entropy_falling.fillna(False)
        short_entry &= recently_peaked.shift(1).fillna(False) & entropy_falling.fillna(False)

    if require_rma_slope_align:
        long_entry &= (slope > 0)
        short_entry &= (slope < 0)

    long_exit = (p_bull <= p_exit) | ((dH > 0) & (T < 0.45))
    short_exit = (p_bear <= p_exit) | ((dH > 0) & (T < 0.45))

    d["long_entry"] = long_entry.fillna(False)
    d["long_exit"] = long_exit.fillna(False)
    d["short_entry"] = short_entry.fillna(False)
    d["short_exit"] = short_exit.fillna(False)

    # build a position state (1 long, -1 short, 0 flat), causal sequential logic
    pos = np.zeros(len(d), dtype=int)
    state = 0
    for i in range(len(d)):
        if state == 0:
            if d["long_entry"].iat[i]:
                state = 1
            elif d["short_entry"].iat[i]:
                state = -1
        elif state == 1:
            if d["long_exit"].iat[i]:
                state = 0
        elif state == -1:
            if d["short_exit"].iat[i]:
                state = 0
        pos[i] = state
    d["position"] = pos

    return d
def plot_quant_physics_dashboard(df_sig: pd.DataFrame, title: str = "Quant-Physics Stack (P2+P4+P1+O5)") -> go.Figure:
    """
    Plotly multi-panel:
    1) Price + Relativistic TD MA + entry/exit markers
    2) Entropy Level
    3) Tunneling Probability
    4) Bayesian Bull/Bear Probabilities
    """
    df = df_sig.copy()

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.18, 0.18, 0.19],
        subplot_titles=(
            "Price Path + Relativistic Time-Dilation MA",
            "Thermodynamic Entropy (Shannon) — Disorder Level",
            "Quantum Tunneling Probability — Barrier Crossing Likelihood",
            "Recursive Bayesian Trend Filter — Bull/Bear Posterior"
        )
    )

    # --- Row 1: price + MA ---
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df.get("Open", df["Close"]),
        high=df.get("High", df["Close"]),
        low=df.get("Low", df["Close"]),
        close=df["Close"],
        name="Price"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Relativistic_TD_MA"],
        mode="lines",
        name="Relativistic_TD_MA (P2)"
    ), row=1, col=1)

    # markers
    le = df.index[df["long_entry"]]
    lx = df.index[df["long_exit"]]
    se = df.index[df["short_entry"]]
    sx = df.index[df["short_exit"]]

    fig.add_trace(go.Scatter(
        x=le, y=df.loc[le, "Close"],
        mode="markers", name="Long Entry",
        marker=dict(symbol="triangle-up", size=10)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=lx, y=df.loc[lx, "Close"],
        mode="markers", name="Long Exit",
        marker=dict(symbol="x", size=10)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=se, y=df.loc[se, "Close"],
        mode="markers", name="Short Entry",
        marker=dict(symbol="triangle-down", size=10)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sx, y=df.loc[sx, "Close"],
        mode="markers", name="Short Exit",
        marker=dict(symbol="x", size=10)
    ), row=1, col=1)

    # --- Row 2: entropy ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Entropy_Level"],
        mode="lines", name="Entropy_Level (P4)"
    ), row=2, col=1)

    # --- Row 3: tunneling ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Tunneling_Prob"],
        mode="lines", name="Tunneling_Prob (P1)"
    ), row=3, col=1)

    # --- Row 4: Bayesian probs ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Bayes_Bull_Prob"],
        mode="lines", name="P(Bull) (O5)"
    ), row=4, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df["Bayes_Bear_Prob"],
        mode="lines", name="P(Bear) (O5)"
    ), row=4, col=1)

    # axes labels (physics terms)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Entropy Level", row=2, col=1)
    fig.update_yaxes(title_text="Tunneling Prob", row=3, col=1, range=[0, 1])
    fig.update_yaxes(title_text="Posterior Prob", row=4, col=1, range=[0, 1])

    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        height=950,
        legend_orientation="h",
        legend_yanchor="bottom",
        legend_y=1.02,
        legend_xanchor="left",
        legend_x=0
    )

    return fig
