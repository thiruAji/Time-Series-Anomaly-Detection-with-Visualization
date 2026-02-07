# data_generation.py
import numpy as np
import pandas as pd

def generate_data(n_steps=3000, n_features=3):
    t = np.arange(n_steps)
    data = []

    for i in range(n_features):
        series = (
            np.sin(2 * np.pi * t / 50) +          # seasonality
            0.001 * t +                            # trend
            np.random.normal(0, 0.2, n_steps)     # noise
        )
        data.append(series)

    data = np.vstack(data).T

    # Inject anomalies
    anomaly_idx = np.random.choice(n_steps, size=30, replace=False)
    data[anomaly_idx] += np.random.normal(5, 1, data[anomaly_idx].shape)

    return pd.DataFrame(data), anomaly_idx
