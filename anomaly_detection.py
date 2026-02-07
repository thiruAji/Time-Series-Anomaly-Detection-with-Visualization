# anomaly_detection.py
import numpy as np

def detect_anomalies(y_true, y_pred, sigma=3):
    """Detect anomalies using residual-based method
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        sigma: Multiplier for standard deviation threshold (default: 3)
    
    Returns:
        Indices of detected anomalies
    """
    residuals = np.abs(y_true - y_pred)
    threshold = residuals.mean() + sigma * residuals.std()
    anomalies = np.where(residuals > threshold)[0]
    return anomalies, threshold
