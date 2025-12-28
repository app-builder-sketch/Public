"""
QUANTUM-GRAVITATIONAL TRADING SYSTEM
Author: Quantitative Financial Physicist
Version: 3.0-QUANT-PHYSICS-ENGINE
Date: 2024

Implements General Relativity, Quantum Mechanics, and Hyperbolic Geometry
for Financial Market Analysis
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.stats import gaussian_kde, ks_2samp
from scipy.integrate import quad
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta

# ============================================================================
# PHYSICAL CONSTANTS (MARKET-ADAPTED)
# ============================================================================
class PhysicsConstants:
    """Market-adapted physical constants"""
    C_MARKET = 299792458  # Speed of light (maximum information speed proxy)
    G_MARKET = 6.67430e-11 * 1e12  # Gravitational constant (scaled for financial values)
    HBAR = 1.0545718e-34  # Reduced Planck constant
    K_BOLTZMANN = 1.380649e-23  # Boltzmann constant
    EULER_MASCHERONI = 0.5772156649  # Euler-Mascheroni constant

# ============================================================================
# 1. GRAVITATIONAL LENSING TREND LINES (P5)
# ============================================================================
class GravitationalLensing:
    """
    Implements Concept P5: General Relativity spacetime curvature
    Treats price pivots as gravitational masses that warp coordinate space
    """
    
    @staticmethod
    def find_pivots(high, low, window=20, threshold=0.03):
        """Find local maxima and minima using adaptive thresholding"""
        n = len(high)
        high_pivots = np.zeros(n)
        low_pivots = np.zeros(n)
        
        for i in range(window, n-window):
            # Look for local highs
            local_high = high[i-window:i+window+1].max()
            local_low = low[i-window:i+window+1].min()
            
            if high[i] == local_high:
                price_move = (local_high - local_low) / local_low
                if price_move > threshold:
                    high_pivots[i] = 1
            
            if low[i] == local_low:
                price_move = (local_high - local_low) / local_low
                if price_move > threshold:
                    low_pivots[i] = 1
        
        return high_pivots, low_pivots
    
    @staticmethod
    def calculate_mass(high, low, volume, high_pivots, low_pivots):
        """
        Calculate gravitational mass: M = Volume × Price Delta
        Based on mass-energy equivalence: E = mc²
        """
        n = len(high)
        price_delta = high - low
        
        # Normalize components
        volume_norm = volume / (np.std(volume) + 1e-10)
        delta_norm = price_delta / (np.std(price_delta) + 1e-10)
        
        # Calculate mass with damping factor for recentness
        mass = volume_norm * delta_norm * (high_pivots + low_pivots)
        
        # Apply time decay: more recent pivots have stronger influence
        time_decay = np.exp(-np.arange(n)[::-1] / 252)  # Annual decay
        mass *= time_decay
        
        return mass
    
    @staticmethod
    def spacetime_curvature(prices, pivot_prices, pivot_masses, pivot_types, 
                           c=PhysicsConstants.C_MARKET):
        """
        Calculate warped price coordinates using Schwarzschild metric approximation
        dτ² = (1 - rs/r)dt² - (1 - rs/r)⁻¹dr²
        """
        n = len(prices)
        curvature = np.zeros(n)
        
        # Filter valid pivots
        valid_indices = np.where((pivot_masses > 0) & (~np.isnan(pivot_prices)))[0]
        
        if len(valid_indices) == 0:
            return prices
        
        for idx in valid_indices:
            pivot_price = pivot_prices[idx]
            mass = pivot_masses[idx]
            pivot_type = pivot_types[idx]
            
            if mass <= 0:
                continue
            
            # Calculate Schwarzschild radius: rs = 2GM/c²
            rs = 2 * PhysicsConstants.G_MARKET * mass / (c**2)
            
            # Distance from pivot (price space)
            distance = np.abs(prices - pivot_price)
            
            # Avoid division by zero
            distance = np.maximum(distance, 1e-10)
            
            # Calculate curvature effect using geodesic equation
            # Δx = rs/2 * log(1 - rs/r) + higher order terms
            with np.errstate(invalid='ignore'):
                warp = rs / 2 * np.log(1 - rs / distance)
                warp = np.nan_to_num(warp, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply direction based on pivot type
            if pivot_type == 'support':
                curvature -= warp  # Support pulls price downward
            elif pivot_type == 'resistance':
                curvature += warp  # Resistance pushes price upward
        
        # Apply smoothing to curvature
        curvature = pd.Series(curvature).rolling(window=5, center=True).mean().values
        
        # Warp the original prices
        warped_prices = prices + curvature
        
        return warped_prices, curvature

# ============================================================================
# 2. QUANTUM TUNNELING PROBABILITY OSCILLATOR (P1)
# ============================================================================
class QuantumTunneling:
    """
    Implements Concept P1: Schrödinger equation for price barriers
    Calculates probability of price tunneling through support/resistance
    """
    
    @staticmethod
    def tunneling_probability(E, V, a, m=1.0):
        """
        Solve 1D Schrödinger equation for rectangular barrier
        T = 1 / [1 + (V² sinh²(κa)) / (4E(V-E))]
        where κ = √[2m(V-E)]/ħ
        """
        # Ensure valid energy range
        if V <= E:
            return 1.0  # No barrier
        
        # Calculate wavenumber inside barrier
        kappa = np.sqrt(2 * m * (V - E)) / PhysicsConstants.HBAR
        
        # Avoid overflow in sinh
        kappa_a = kappa * a
        if kappa_a > 100:  # Very large barrier
            return 0.0
        
        sinh_term = np.sinh(kappa_a)
        tunneling = 1 / (1 + (V**2 * sinh_term**2) / (4 * E * (V - E)))
        
        return np.clip(tunneling, 0, 1)
    
    @staticmethod
    def calculate_barriers(prices, high_pivots, low_pivots):
        """Identify support and resistance barriers"""
        n = len(prices)
        supports = []
        resistances = []
        
        # Find support levels (low pivots)
        low_idx = np.where(low_pivots == 1)[0]
        for idx in low_idx:
            if idx < n:
                supports.append({
                    'price': prices[idx],
                    'index': idx,
                    'strength': 1.0
                })
        
        # Find resistance levels (high pivots)
        high_idx = np.where(high_pivots == 1)[0]
        for idx in high_idx:
            if idx < n:
                resistances.append({
                    'price': prices[idx],
                    'index': idx,
                    'strength': 1.0
                })
        
        return supports, resistances
    
    @staticmethod
    def calculate_oscillator(prices, supports, resistances, volatility):
        """
        Calculate quantum tunneling oscillator
        Positive = tunneling through resistance likely
        Negative = tunneling through support likely
        """
        n = len(prices)
        oscillator = np.zeros(n)
        prob_support = np.zeros(n)
        prob_resistance = np.zeros(n)
        
        # Calculate current price "energy" (kinetic energy analogy)
        # E = 0.5 * m * v² where v = normalized volatility
        m = 1.0  # Effective mass of price particle
        price_energy = 0.5 * m * (volatility ** 2)
        
        for i in range(n):
            current_price = prices[i]
            
            # Find nearest support and resistance
            nearest_support = None
            nearest_resistance = None
            
            # Support barriers (below current price)
            if len(supports) > 0:
                valid_supports = [s for s in supports if s['price'] < current_price]
                if valid_supports:
                    nearest_support = min(valid_supports, 
                                        key=lambda x: abs(x['price'] - current_price))
            
            # Resistance barriers (above current price)
            if len(resistances) > 0:
                valid_resistances = [r for r in resistances if r['price'] > current_price]
                if valid_resistances:
                    nearest_resistance = min(valid_resistances,
                                           key=lambda x: abs(x['price'] - current_price))
            
            # Calculate tunneling probabilities
            if nearest_support:
                barrier_height = abs(nearest_support['price'] - current_price) / current_price
                V0 = max(barrier_height * price_energy * 100, price_energy * 1.01)
                barrier_width = volatility / 50  # Uncertainty principle proxy
                prob_support[i] = QuantumTunneling.tunneling_probability(
                    price_energy, V0, barrier_width, m
                )
            
            if nearest_resistance:
                barrier_height = abs(nearest_resistance['price'] - current_price) / current_price
                V0 = max(barrier_height * price_energy * 100, price_energy * 1.01)
                barrier_width = volatility / 50
                prob_resistance[i] = QuantumTunneling.tunneling_probability(
                    price_energy, V0, barrier_width, m
                )
            
            # Net oscillator
            oscillator[i] = prob_resistance[i] - prob_support[i]
        
        return oscillator, prob_support, prob_resistance

# ============================================================================
# 3. HYPERBOLIC PATH CURVATURE (C3)
# ============================================================================
class HyperbolicGeometry:
    """
    Implements Concept C3: Non-Euclidean geometry for trend analysis
    Poincaré disk model for hyperbolic space
    """
    
    @staticmethod
    def poincare_distance(z1, z2):
        """
        Calculate hyperbolic distance in Poincaré disk
        d(z1, z2) = acosh(1 + 2 * |z1 - z2|² / ((1 - |z1|²) * (1 - |z2|²)))
        """
        num = 2 * np.abs(z1 - z2) ** 2
        denom = (1 - np.abs(z1) ** 2) * (1 - np.abs(z2) ** 2)
        
        # Handle edge cases
        if denom <= 0:
            return np.inf
        
        return np.arccosh(1 + num / denom)
    
    @staticmethod
    def calculate_curvature(prices, time_indices, window=30):
        """
        Calculate hyperbolic curvature of price path
        Higher curvature = more acceleration/deceleration
        """
        n = len(prices)
        curvature = np.zeros(n)
        geodesic_deviation = np.zeros(n)
        
        # Normalize to [0, 1] for Poincaré disk
        price_min, price_max = np.min(prices), np.max(prices)
        price_range = price_max - price_min + 1e-10
        norm_prices = (prices - price_min) / price_range
        
        # Normalize time
        time_min, time_max = np.min(time_indices), np.max(time_indices)
        time_range = time_max - time_min + 1e-10
        norm_times = (time_indices - time_min) / time_range
        
        # Map to complex plane (Poincaré disk coordinates)
        z = norm_prices + 1j * norm_times
        
        for i in range(window, n-window):
            # Extract window
            start = i - window
            end = i + window
            
            window_z = z[start:end]
            
            if len(window_z) < 3:
                curvature[i] = 0
                geodesic_deviation[i] = 0
                continue
            
            # Calculate total path length in hyperbolic space
            path_length = 0
            for j in range(1, len(window_z)):
                path_length += HyperbolicGeometry.poincare_distance(
                    window_z[j-1], window_z[j]
                )
            
            # Calculate chord length (straight line in hyperbolic space)
            chord_length = HyperbolicGeometry.poincare_distance(
                window_z[0], window_z[-1]
            )
            
            # Curvature = path_length / chord_length - 1
            if chord_length > 1e-10:
                curvature[i] = path_length / chord_length - 1
                geodesic_deviation[i] = path_length - chord_length
            else:
                curvature[i] = 0
                geodesic_deviation[i] = 0
        
        # Smooth curvature
        curvature = pd.Series(curvature).rolling(window=10, center=True).mean().values
        geodesic_deviation = pd.Series(geodesic_deviation).rolling(window=10, center=True).mean().values
        
        return curvature, geodesic_deviation
    
    @staticmethod
    def geodesic_acceleration(prices, curvature, window=20):
        """
        Calculate acceleration along geodesic
        Positive = trend acceleration, Negative = deceleration
        """
        n = len(prices)
        acceleration = np.zeros(n)
        
        for i in range(window, n-window):
            # Calculate second derivative of curvature
            if i >= 2 and i < n-2:
                # Second order finite difference
                d2c = curvature[i-2] - 2*curvature[i] + curvature[i+2]
                acceleration[i] = d2c / 4
        
        # Normalize acceleration
        acc_std = np.std(acceleration[acceleration != 0])
        if acc_std > 0:
            acceleration = acceleration / acc_std
        
        return acceleration

# ============================================================================
# 4. RIEMANN ZETA FUNCTION OSCILLATOR (C1)
# ============================================================================
class RiemannZetaOscillator:
    """
    Implements Concept C1: Riemann Zeta critical line analysis
    Uses Dirichlet series approximation for market regime detection
    """
    
    @staticmethod
    def dirichlet_series(s, n_terms=100):
        """Truncated Dirichlet series approximation of ζ(s)"""
        terms = np.arange(1, n_terms + 1)
        return np.sum(1 / (terms ** s))
    
    @staticmethod
    def zeta_oscillator(prices, n_terms=50):
        """
        Map price action to Riemann Zeta function critical line (Re=0.5)
        Zero-crossings indicate potential trend changes
        """
        n = len(prices)
        oscillator = np.zeros(n)
        
        # Normalize prices to complex domain
        price_min, price_max = np.min(prices), np.max(prices)
        price_range = price_max - price_min + 1e-10
        
        for i in range(n_terms, n):
            # Extract window
            window = prices[i-n_terms:i]
            
            # Normalize to [0, 1]
            norm_window = (window - price_min) / price_range
            
            # Calculate mean and std for complex argument
            mu = np.mean(norm_window)
            sigma = np.std(norm_window) + 1e-10
            
            # Map to critical line: s = 0.5 + it
            # where t is related to market volatility/structure
            t = 14.134725 * (sigma / mu) if mu > 0 else 14.134725  # First non-trivial zero
            
            s = 0.5 + 1j * t
            
            # Calculate zeta value
            zeta_val = RiemannZetaOscillator.dirichlet_series(s, n_terms=30)
            
            # Oscillator is imaginary part (phase information)
            oscillator[i] = np.imag(zeta_val)
        
        # Normalize oscillator
        osc_std = np.std(oscillator[oscillator != 0])
        if osc_std > 0:
            oscillator = oscillator / osc_std
        
        return oscillator
    
    @staticmethod
    def zero_crossings(oscillator, threshold=0.1):
        """Detect zero crossings in zeta oscillator"""
        n = len(oscillator)
        crossings = np.zeros(n)
        
        for i in range(1, n):
            if oscillator[i-1] <= threshold and oscillator[i] > threshold:
                crossings[i] = 1  # Positive crossing
            elif oscillator[i-1] >= -threshold and oscillator[i] < -threshold:
                crossings[i] = -1  # Negative crossing
        
        return crossings

# ============================================================================
# 5. LYAPUNOV EXPONENT STABILITY MAP (C5)
# ============================================================================
class LyapunovAnalyzer:
    """
    Implements Concept C5: Chaos theory for market stability
    Calculates local Lyapunov exponents to detect chaotic regimes
    """
    
    @staticmethod
    def embed_time_series(series, embedding_dim=3, tau=1):
        """Time delay embedding for phase space reconstruction"""
        n = len(series)
        embedded = []
        
        for i in range((embedding_dim-1)*tau, n):
            point = []
            for j in range(embedding_dim):
                point.append(series[i - j*tau])
            embedded.append(point)
        
        return np.array(embedded)
    
    @staticmethod
    def local_lyapunov(series, window=100, embedding_dim=3, tau=1):
        """
        Calculate local largest Lyapunov exponent
        Positive = chaotic, Negative = stable
        """
        n = len(series)
        exponents = np.zeros(n)
        
        for i in range(window, n-window):
            # Extract window
            window_data = series[i-window:i+window]
            
            if len(window_data) < embedding_dim * tau + 10:
                exponents[i] = 0
                continue
            
            # Phase space reconstruction
            embedded = LyapunovAnalyzer.embed_time_series(window_data, embedding_dim, tau)
            
            if len(embedded) < 10:
                exponents[i] = 0
                continue
            
            # Calculate nearest neighbor distances over time
            m = len(embedded)
            distances = []
            
            # Reference trajectory
            ref_idx = m // 2
            
            # Find nearest neighbor to reference point
            ref_point = embedded[ref_idx]
            
            # Calculate distances to all other points
            dists = np.linalg.norm(embedded - ref_point, axis=1)
            dists[ref_idx] = np.inf  # Exclude reference itself
            
            # Find nearest neighbor
            nn_idx = np.argmin(dists)
            
            # Track divergence
            divergence = []
            max_steps = min(20, m - max(ref_idx, nn_idx))
            
            for k in range(1, max_steps):
                d0 = np.linalg.norm(embedded[ref_idx] - embedded[nn_idx])
                dk = np.linalg.norm(embedded[ref_idx+k] - embedded[nn_idx+k])
                
                if d0 > 0 and dk > 0:
                    divergence.append(np.log(dk / d0))
            
            if len(divergence) > 0:
                # Lyapunov exponent is average divergence rate
                exponents[i] = np.mean(divergence) / max_steps
            else:
                exponents[i] = 0
        
        # Smooth exponents
        exponents = pd.Series(exponents).rolling(window=20, center=True).mean().values
        
        return exponents
    
    @staticmethod
    def chaos_regime(exponents, threshold=0.001):
        """Classify market regime based on Lyapunov exponent"""
        n = len(exponents)
        regime = np.zeros(n)
        
        for i in range(n):
            if exponents[i] > threshold:
                regime[i] = 2  # Chaotic
            elif exponents[i] < -threshold:
                regime[i] = 0  # Stable
            else:
                regime[i] = 1  # Transitional
        
        return regime

# ============================================================================
# 6. FLUID DYNAMICS FLOW REGIME (P3)
# ============================================================================
class FluidDynamics:
    """
    Implements Concept P3: Laminar vs Turbulent flow detection
    Calculates market Reynolds number for regime classification
    """
    
    @staticmethod
    def reynolds_number(prices, volume, window=20):
        """
        Calculate financial Reynolds number
        Re = (Density * Velocity * Length) / Viscosity
        
        Financial analogy:
        - Density = Trading intensity (volume/price)
        - Velocity = Price change rate
        - Length = Price range
        - Viscosity = Market friction (volatility)
        """
        n = len(prices)
        reynolds = np.zeros(n)
        
        returns = np.diff(prices, prepend=prices[0]) / prices
        
        for i in range(window, n):
            # Extract window
            price_window = prices[i-window:i]
            vol_window = volume[i-window:i]
            ret_window = returns[i-window:i]
            
            # Calculate components
            # Density ~ normalized volume
            density = np.mean(vol_window) / (np.mean(price_window) + 1e-10)
            
            # Velocity ~ absolute returns
            velocity = np.std(ret_window) * 100  # Percentage
            
            # Length ~ price range normalized
            length = (np.max(price_window) - np.min(price_window)) / np.mean(price_window)
            
            # Viscosity ~ inverse of trend smoothness
            # Smooth trend = low viscosity, choppy = high viscosity
            autocorr = np.corrcoef(price_window[:-1], price_window[1:])[0, 1]
            viscosity = 1 / (abs(autocorr) + 0.1)  # Avoid division by zero
            
            # Reynolds number
            if viscosity > 0:
                reynolds[i] = (density * velocity * length) / viscosity
            else:
                reynolds[i] = 0
        
        # Normalize
        re_std = np.std(reynolds[reynolds != 0])
        if re_std > 0:
            reynolds = reynolds / re_std
        
        return reynolds
    
    @staticmethod
    def flow_regime(reynolds, laminar_thresh=0.5, turbulent_thresh=1.5):
        """Classify flow regime based on Reynolds number"""
        n = len(reynolds)
        regime = np.zeros(n)
        
        for i in range(n):
            if abs(reynolds[i]) < laminar_thresh:
                regime[i] = 0  # Laminar (trending)
            elif abs(reynolds[i]) > turbulent_thresh:
                regime[i] = 2  # Turbulent (choppy)
            else:
                regime[i] = 1  # Transitional
        
        return regime

# ============================================================================
# 7. SIGNAL GENERATION ENGINE
# ============================================================================
class SignalGenerator:
    """Generates trading signals based on physics indicators"""
    
    def __init__(self):
        self.signals = {}
        self.confidence = {}
        
    def generate_signals(self, indicators, params=None):
        """
        Generate Buy/Sell signals based on physics convergence
        """
        if params is None:
            params = {
                'tunnel_threshold': 0.7,
                'curvature_threshold': 0.2,
                'lyapunov_threshold': 0.001,
                'reynolds_threshold': 1.0
            }
        
        n = len(indicators['close'])
        signals = np.zeros(n)
        confidence = np.zeros(n)
        
        for i in range(50, n):  # Skip warm-up period
            signal_score = 0
            conf_score = 0
            vote_count = 0
            
            # 1. Quantum Tunneling Signal
            tunnel_signal = 0
            if 'tunneling_oscillator' in indicators:
                osc = indicators['tunneling_oscillator'][i]
                if osc > params['tunnel_threshold']:
                    tunnel_signal = 1
                    conf_score += abs(osc)
                    vote_count += 1
                elif osc < -params['tunnel_threshold']:
                    tunnel_signal = -1
                    conf_score += abs(osc)
                    vote_count += 1
            
            # 2. Curvature Signal
            curvature_signal = 0
            if 'hyperbolic_curvature' in indicators:
                curv = indicators['hyperbolic_curvature'][i]
                if abs(curv) < params['curvature_threshold']:
                    # Low curvature supports trend continuation
                    curvature_signal = tunnel_signal  # Align with tunneling
                    conf_score += (1 - abs(curv) / params['curvature_threshold'])
                    vote_count += 1
            
            # 3. Lyapunov Stability Signal
            stability_signal = 0
            if 'lyapunov_exponents' in indicators:
                lyap = indicators['lyapunov_exponents'][i]
                if lyap < -params['lyapunov_threshold']:
                    # Stable regime - trust trend signals
                    stability_signal = tunnel_signal
                    conf_score += 0.5
                    vote_count += 1
            
            # 4. Reynolds Flow Signal
            flow_signal = 0
            if 'reynolds_number' in indicators:
                reynolds = indicators['reynolds_number'][i]
                if abs(reynolds) < params['reynolds_threshold']:
                    # Laminar flow - good for trending strategies
                    flow_signal = tunnel_signal
                    conf_score += 0.5
                    vote_count += 1
            
            # 5. Zeta Zero-Crossing Signal
            zeta_signal = 0
            if 'zeta_crossings' in indicators:
                if indicators['zeta_crossings'][i] != 0:
                    zeta_signal = indicators['zeta_crossings'][i]
                    conf_score += 0.3
                    vote_count += 1
            
            # Combine signals (weighted voting)
            if vote_count > 0:
                signal_score = (tunnel_signal * 2 + curvature_signal + 
                              stability_signal + flow_signal + zeta_signal)
                
                # Normalize
                signal_score = signal_score / (vote_count + 2)  # +2 for tunnel double weight
                
                # Determine final signal
                if signal_score > 0.3:
                    signals[i] = 1  # Buy
                    confidence[i] = conf_score / vote_count if vote_count > 0 else 0
                elif signal_score < -0.3:
                    signals[i] = -1  # Sell
                    confidence[i] = conf_score / vote_count if vote_count > 0 else 0
        
        return signals, confidence

# ============================================================================
# 8. MAIN PHYSICS ENGINE
# ============================================================================
class PhysicsTradingEngine:
    """Main engine integrating all physics concepts"""
    
    def __init__(self):
        self.gravity = GravitationalLensing()
        self.quantum = QuantumTunneling()
        self.hyperbolic = HyperbolicGeometry()
        self.zeta = RiemannZetaOscillator()
        self.lyapunov = LyapunovAnalyzer()
        self.fluid = FluidDynamics()
        self.signal_gen = SignalGenerator()
        
    def calculate_all_indicators(self, df):
        """
        Calculate all physics-based indicators
        df must contain: 'open', 'high', 'low', 'close', 'volume'
        """
        # Extract data
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        volume = df['volume'].values
        n = len(close)
        
        # Time indices for hyperbolic geometry
        time_indices = np.arange(n)
        
        # Calculate returns and volatility
        returns = np.diff(close, prepend=close[0]) / close
        volatility = pd.Series(returns).rolling(window=20).std().values * np.sqrt(252)
        
        indicators = {
            'close': close,
            'high': high,
            'low': low,
            'volume': volume,
            'returns': returns,
            'volatility': volatility
        }
        
        print("Step 1/8: Finding price pivots...")
        # 1. Find pivots
        high_pivots, low_pivots = self.gravity.find_pivots(high, low)
        
        print("Step 2/8: Calculating gravitational masses...")
        # 2. Calculate gravitational masses
        gravitational_mass = self.gravity.calculate_mass(high, low, volume, 
                                                        high_pivots, low_pivots)
        
        # Create pivot arrays
        pivot_prices = np.full(n, np.nan)
        pivot_types = np.full(n, '', dtype=object)
        
        high_idx = np.where(high_pivots == 1)[0]
        low_idx = np.where(low_pivots == 1)[0]
        
        pivot_prices[high_idx] = high[high_idx]
        pivot_types[high_idx] = 'resistance'
        pivot_prices[low_idx] = low[low_idx]
        pivot_types[low_idx] = 'support'
        
        print("Step 3/8: Applying gravitational lensing...")
        # 3. Calculate spacetime curvature
        warped_prices, curvature = self.gravity.spacetime_curvature(
            close, pivot_prices, gravitational_mass, pivot_types
        )
        
        indicators['warped_close'] = warped_prices
        indicators['spacetime_curvature'] = curvature
        indicators['gravitational_mass'] = gravitational_mass
        indicators['pivot_prices'] = pivot_prices
        indicators['pivot_types'] = pivot_types
        
        print("Step 4/8: Calculating quantum tunneling...")
        # 4. Quantum tunneling oscillator
        supports, resistances = self.quantum.calculate_barriers(
            close, high_pivots, low_pivots
        )
        
        tunneling_oscillator, prob_support, prob_resistance = (
            self.quantum.calculate_oscillator(
                close, supports, resistances, volatility
            )
        )
        
        indicators['tunneling_oscillator'] = tunneling_oscillator
        indicators['tunnel_prob_support'] = prob_support
        indicators['tunnel_prob_resistance'] = prob_resistance
        
        print("Step 5/8: Calculating hyperbolic curvature...")
        # 5. Hyperbolic geometry
        hyperbolic_curvature, geodesic_deviation = self.hyperbolic.calculate_curvature(
            close, time_indices
        )
        
        indicators['hyperbolic_curvature'] = hyperbolic_curvature
        indicators['geodesic_deviation'] = geodesic_deviation
        
        print("Step 6/8: Calculating Riemann Zeta oscillator...")
        # 6. Riemann Zeta oscillator
        zeta_oscillator = self.zeta.zeta_oscillator(close)
        zeta_crossings = self.zeta.zero_crossings(zeta_oscillator)
        
        indicators['zeta_oscillator'] = zeta_oscillator
        indicators['zeta_crossings'] = zeta_crossings
        
        print("Step 7/8: Calculating Lyapunov exponents...")
        # 7. Lyapunov exponents
        lyapunov_exponents = self.lyapunov.local_lyapunov(close)
        chaos_regime = self.lyapunov.chaos_regime(lyapunov_exponents)
        
        indicators['lyapunov_exponents'] = lyapunov_exponents
        indicators['chaos_regime'] = chaos_regime
        
        print("Step 8/8: Calculating fluid dynamics...")
        # 8. Fluid dynamics
        reynolds_number = self.fluid.reynolds_number(close, volume)
        flow_regime = self.fluid.flow_regime(reynolds_number)
        
        indicators['reynolds_number'] = reynolds_number
        indicators['flow_regime'] = flow_regime
        
        print("Generating trading signals...")
        # Generate signals
        signals, confidence = self.signal_gen.generate_signals(indicators)
        
        indicators['signals'] = signals
        indicators['signal_confidence'] = confidence
        
        # Create DataFrame with all indicators
        result_df = pd.DataFrame({
            'close': close,
            'warped_close': warped_prices,
            'gravitational_mass': gravitational_mass,
            'tunneling_oscillator': tunneling_oscillator,
            'tunnel_prob_support': prob_support,
            'tunnel_prob_resistance': prob_resistance,
            'hyperbolic_curvature': hyperbolic_curvature,
            'geodesic_deviation': geodesic_deviation,
            'zeta_oscillator': zeta_oscillator,
            'zeta_crossings': zeta_crossings,
            'lyapunov_exponents': lyapunov_exponents,
            'chaos_regime': chaos_regime,
            'reynolds_number': reynolds_number,
            'flow_regime': flow_regime,
            'signals': signals,
            'signal_confidence': confidence
        }, index=df.index)
        
        return result_df

# ============================================================================
# 9. VISUALIZATION ENGINE
# ============================================================================
class PhysicsVisualizer:
    """Creates comprehensive physics-based visualization dashboard"""
    
    @staticmethod
    def create_dashboard(df, indicators, symbol="ASSET"):
        """Create interactive Plotly dashboard"""
        
        # Create subplot layout
        fig = make_subplots(
            rows=5, cols=2,
            subplot_titles=(
                f"{symbol} - Price & Gravitational Lensing",
                "Quantum Tunneling Oscillator",
                "Hyperbolic Space Analysis",
                "Riemann Zeta & Lyapunov Stability",
                "Fluid Dynamics Regime",
                "Signal Confidence & Distribution",
                "Trading Signals Overlay",
                "Physics Convergence Map",
                "Indicator Correlation Matrix",
                "Performance Statistics"
            ),
            vertical_spacing=0.08,
            horizontal_spacing=0.1,
            specs=[
                [{"type": "scatter", "rowspan": 2}, {"type": "scatter"}],
                [None, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "scatter"}, {"type": "histogram"}]
            ],
            row_heights=[0.3, 0.1, 0.2, 0.2, 0.2]
        )
        
        # 1. Price with Gravitational Lensing (Row 1-2, Col 1)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['close'],
                name='Actual Price',
                line=dict(color='blue', width=1),
                opacity=0.7
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=indicators['warped_close'],
                name='Warped Price (Lensed)',
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Add gravitational masses as markers
        mass_indices = np.where(indicators['gravitational_mass'] > 0)[0]
        if len(mass_indices) > 0:
            mass_prices = indicators['close'].iloc[mass_indices]
            mass_values = indicators['gravitational_mass'].iloc[mass_indices]
            
            fig.add_trace(
                go.Scatter(
                    x=df.index[mass_indices],
                    y=mass_prices,
                    mode='markers',
                    name='Gravitational Mass',
                    marker=dict(
                        size=mass_values * 10,
                        color=mass
