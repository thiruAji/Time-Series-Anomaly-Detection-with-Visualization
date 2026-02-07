# data_generation.py
"""
Advanced Synthetic Time Series Data Generator
Generates 5 multivariate features with trend, seasonality, and injected anomalies.
"""
import numpy as np
import pandas as pd

def generate_data(n_steps=1000, n_features=5, anomaly_ratio=0.03, seed=42):
    """
    Generate complex multivariate time series data with known anomalies.
    
    Features:
        1. base: Core signal with seasonality
        2. trend: Linear + quadratic trend component
        3. seasonality: Multiple seasonal patterns (daily, weekly)
        4. momentum: Rate of change / derivative feature
        5. volatility: Rolling standard deviation (volatility clustering)
    
    Args:
        n_steps: Number of time steps
        n_features: Number of features (default 5)
        anomaly_ratio: Fraction of points to inject as anomalies
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame of shape (n_steps, n_features), array of anomaly indices
    """
    np.random.seed(seed)
    t = np.arange(n_steps)
    
    # Feature 1: Base signal with primary seasonality
    base = (
        np.sin(2 * np.pi * t / 50) * 2 +           # Primary cycle (50 steps)
        np.sin(2 * np.pi * t / 200) * 0.5 +        # Secondary cycle (200 steps)
        np.random.normal(0, 0.3, n_steps)          # Noise
    )
    
    # Feature 2: Trend component (linear + slight quadratic)
    trend = (
        0.002 * t +                                # Linear trend
        0.000001 * (t ** 2) +                      # Quadratic acceleration
        np.random.normal(0, 0.1, n_steps)          # Noise
    )
    
    # Feature 3: Seasonality (multiple periods - daily/weekly patterns)
    seasonality = (
        np.sin(2 * np.pi * t / 7) * 1.5 +          # Weekly pattern
        np.sin(2 * np.pi * t / 30) * 0.8 +         # Monthly pattern
        np.cos(2 * np.pi * t / 365) * 0.3 +        # Yearly pattern
        np.random.normal(0, 0.2, n_steps)          # Noise
    )
    
    # Feature 4: Momentum (rate of change of base)
    momentum = np.gradient(base) + np.random.normal(0, 0.1, n_steps)
    
    # Feature 5: Volatility (rolling std of base - volatility clustering)
    window = 10
    volatility = pd.Series(base).rolling(window=window, min_periods=1).std().values
    volatility += np.random.normal(0, 0.05, n_steps)
    
    # Combine into DataFrame
    data = pd.DataFrame({
        'base': base,
        'trend': trend,
        'seasonality': seasonality,
        'momentum': momentum,
        'volatility': volatility
    })
    
    # Inject anomalies at random locations
    n_anomalies = int(n_steps * anomaly_ratio)
    anomaly_indices = np.random.choice(n_steps, size=n_anomalies, replace=False)
    anomaly_indices = np.sort(anomaly_indices)
    
    # Apply anomalies (large deviations in multiple features)
    for idx in anomaly_indices:
        # Random magnitude and direction
        magnitude = np.random.uniform(3, 7) * np.random.choice([-1, 1])
        # Affect multiple features with correlated anomaly
        data.iloc[idx, 0] += magnitude  # base
        data.iloc[idx, 2] += magnitude * 0.5  # seasonality
        data.iloc[idx, 3] += magnitude * 0.3  # momentum
    
    return data, anomaly_indices


def get_ground_truth_labels(n_steps, anomaly_indices):
    """Convert anomaly indices to binary labels array."""
    labels = np.zeros(n_steps, dtype=int)
    labels[anomaly_indices] = 1
    return labels


if __name__ == "__main__":
    # Test generation
    data, anomalies = generate_data(n_steps=500)
    print(f"Generated data shape: {data.shape}")
    print(f"Features: {list(data.columns)}")
    print(f"Anomaly count: {len(anomalies)}")
    print(f"Anomaly indices: {anomalies[:10]}...")
    print(data.describe())
