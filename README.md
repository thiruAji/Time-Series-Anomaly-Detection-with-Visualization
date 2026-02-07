# 📈 Time Series Anomaly Detection

A robust, production-ready web application for detecting anomalies in time-series data using statistical methods (Z-Score/Residual Analysis).

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 🚀 Features

- **Real-time Visualization**: Interactive charts using Chart.js to display Actual vs. Predicted values.
- **Anomaly Detection**: Automatically flags data points that deviate significantly from predictions.
- **CSV Upload**: Support for custom datasets via CSV upload.
- **Production Server**: Uses `waitress` for stable, thread-safe deployment.
- **API Support**: RESTful endpoints for integration with other systems.

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/time-series-anomaly-detection.git
    cd time-series-anomaly-detection
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🏃 Usage

1.  **Start the server**:
    ```bash
    python server.py
    ```
    *The server will start on `http://0.0.0.0:5001`*

2.  **Open in Browser**:
    Navigate to `http://localhost:5001`

3.  **Analyze Data**:
    *   Click **"Load Demo Data"** to see it in action.
    *   Or click **"📂 Upload CSV File"** to analyze your own data.
    *   Click **"Detect Anomalies"** to see results.

## 📂 Project Structure

```
├── app.py                 # Flask application entry point & routes
├── server.py              # Production WSGI server functionality
├── anomaly_detection.py   # Core logic for statistical anomaly detection
├── data_generation.py     # Synthetic data generator
├── model.py               # LSTM Model definition (PyTorch)
├── templates/
│   └── index.html         # Frontend UI (HTML/JS/Chart.js)
└── requirements.txt       # Project dependencies
```

## 🧠 Algorithmic Details

- **Detection Method**: Residual Analysis
- **Threshold Calculation**: `Mean(Residuals) + (Sigma * StdDev(Residuals))`
- **Model**: LSTM (Long Short-Term Memory) network for robust time-series forecasting.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.