#!/usr/bin/env python
# app.py
from flask import Flask, jsonify, request, render_template_string
import numpy as np
import torch
import os
from model import LSTMModel
from data_generation import generate_data
from anomaly_detection import detect_anomalies
from evaluation import evaluate_model

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global data cache
_data_cache = None

def get_data():
    """Lazy load data"""
    global _data_cache
    if _data_cache is None:
        data_df, _ = generate_data(n_steps=1000, n_features=3)
        _data_cache = data_df.values
    return _data_cache

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Time Series Anomaly Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 30px auto;
            background: #f5f5f5;
            padding: 20px;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        .tabs {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            border-bottom: 1px solid #ddd;
        }
        .tab-btn {
            background: #f0f0f0;
            border: none;
            padding: 10px 20px;
            cursor: pointer;
            border-radius: 4px 4px 0 0;
            font-weight: bold;
        }
        .tab-btn.active {
            background: #007bff;
            color: white;
        }
        .tab-content {
            display: none;
            padding: 20px 0;
        }
        .tab-content.active {
            display: block;
        }
        .form-group {
            margin: 15px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        textarea, input {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
            box-sizing: border-box;
            font-family: monospace;
            font-size: 14px;
        }
        textarea {
            min-height: 100px;
            resize: vertical;
        }
        button {
            background: #007bff;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-top: 10px;
            font-weight: bold;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .result {
            background: #f9f9f9;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 4px;
            margin-top: 20px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .result pre {
            background: #333;
            color: #0f0;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 13px;
            line-height: 1.4;
            max-height: 300px;
            overflow-y: auto;
        }
        .chart-container {
            position: relative;
            height: 400px;
            margin: 20px 0;
            display: none;
        }
        .chart-container.show {
            display: block;
        }
        .status {
            margin-top: 10px;
            padding: 12px;
            border-radius: 4px;
            display: none;
        }
        .status.show {
            display: block;
        }
        .status.success {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        .metrics {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin: 15px 0;
        }
        .metric-card {
            background: #f0f0f0;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Time Series Anomaly Detection with Visualization</h1>
        
        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('detect')">Quick Detect</button>
            <button class="tab-btn" onclick="switchTab('train')">Train & Predict</button>
            <button class="tab-btn" onclick="switchTab('data')">Sample Data</button>
        </div>

        <!-- Quick Detect Tab -->
        <div id="detect" class="tab-content active">
            <h2>Quick Anomaly Detection</h2>
            <div class="form-group">
                <label for="y_true">Actual Values (comma-separated):</label>
                <textarea id="y_true" placeholder="Example: 1, 2, 3, 10, 5, 6">1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 8, 7, 6, 5</textarea>
            </div>

            <div class="form-group">
                <label for="y_pred">Predicted Values (comma-separated):</label>
                <textarea id="y_pred" placeholder="Example: 1.1, 2.1, 2.9, 5, 5.2, 6.1">1.1, 2.2, 3.1, 3.9, 5.1, 6.1, 7.2, 8.1, 9.1, 10.1, 11, 8.2, 7.1, 6.2, 5.1</textarea>
            </div>

            <div class="form-group">
                <label for="sigma">Sensitivity (sigma - higher = stricter):</label>
                <input type="number" id="sigma" value="3" step="0.1" min="0.1">
            </div>

            <button onclick="detectAnomalies()">Detect Anomalies</button>

            <div id="detect-status" class="status"></div>
            
            <div class="chart-container" id="detectChart">
                <canvas id="detectCanvas"></canvas>
            </div>

            <div class="metrics" id="detectMetrics" style="display: none;">
                <div class="metric-card">
                    <div style="color: #666;">MAE</div>
                    <div class="metric-value" id="detectMAE">-</div>
                </div>
                <div class="metric-card">
                    <div style="color: #666;">RMSE</div>
                    <div class="metric-value" id="detectRMSE">-</div>
                </div>
                <div class="metric-card">
                    <div style="color: #666;">Anomalies Found</div>
                    <div class="metric-value" id="detectCount">-</div>
                </div>
                <div class="metric-card">
                    <div style="color: #666;">Threshold</div>
                    <div class="metric-value" id="detectThreshold">-</div>
                </div>
            </div>

            <div id="detect-result" class="result">
                <h3>Raw Results:</h3>
                <pre id="detectContent"></pre>
            </div>
        </div>

        <!-- Train & Predict Tab -->
        <div id="train" class="tab-content">
            <h2>Full Training Pipeline</h2>
            <p>Input time series data, automatically finds best model configuration using NAS, trains it, and detects anomalies.</p>
            
            <div class="form-group">
                <label for="train_data">Time Series Data (comma-separated):</label>
                <textarea id="train_data" placeholder="Example: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 5">1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 9, 8, 7, 6, 5, 4, 3, 2, 1</textarea>
            </div>

            <div class="form-group">
                <label for="train_sigma">Sensitivity:</label>
                <input type="number" id="train_sigma" value="3" step="0.1" min="0.1">
            </div>

            <button onclick="trainPredict()" id="trainBtn">Train & Predict</button>

            <div id="train-status" class="status"></div>

            <div class="chart-container" id="trainChart">
                <canvas id="trainCanvas"></canvas>
            </div>

            <div class="metrics" id="trainMetrics" style="display: none;">
                <div class="metric-card">
                    <div style="color: #666;">Layers (NAS)</div>
                    <div class="metric-value" id="trainLayers">-</div>
                </div>
                <div class="metric-card">
                    <div style="color: #666;">Hidden Units (NAS)</div>
                    <div class="metric-value" id="trainHidden">-</div>
                </div>
                <div class="metric-card">
                    <div style="color: #666;">MAE</div>
                    <div class="metric-value" id="trainMAE">-</div>
                </div>
                <div class="metric-card">
                    <div style="color: #666;">RMSE</div>
                    <div class="metric-value" id="trainRMSE">-</div>
                </div>
            </div>

            <div id="train-result" class="result">
                <h3>Training Details:</h3>
                <pre id="trainContent"></pre>
            </div>
        </div>

        <!-- Sample Data Tab -->
        <div id="data" class="tab-content">
            <h2>Sample Dataset</h2>
            <button onclick="loadSampleData()">Load Sample Data</button>
            
            <div id="data-status" class="status"></div>

            <div class="chart-container" id="dataChart">
                <canvas id="dataCanvas"></canvas>
            </div>
            
            <div id="data-result" class="result">
                <h3>Data Info:</h3>
                <pre id="dataContent"></pre>
            </div>
        </div>
    </div>

    <script>
        let detectChartInstance = null;
        let trainChartInstance = null;
        let dataChartInstance = null;

        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        function parseValues(str) {
            return str.split(',').map(x => parseFloat(x.trim())).filter(x => !isNaN(x));
        }

        function detectAnomalies() {
            const statusEl = document.getElementById('detect-status');
            const chartEl = document.getElementById('detectChart');
            
            try {
                const y_true = parseValues(document.getElementById('y_true').value);
                const y_pred = parseValues(document.getElementById('y_pred').value);
                const sigma = parseFloat(document.getElementById('sigma').value);

                if (y_true.length === 0 || y_pred.length === 0) {
                    throw new Error('Please enter valid values');
                }

                statusEl.textContent = '⏳ Processing...';
                statusEl.className = 'status show info';

                fetch('/detect', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ y_true, y_pred, sigma })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) throw new Error(data.error);
                    
                    statusEl.textContent = '✅ Success!';
                    statusEl.className = 'status show success';
                    
                    // Display metrics
                    document.getElementById('detectMetrics').style.display = 'grid';
                    document.getElementById('detectMAE').textContent = data.metrics.mae.toFixed(4);
                    document.getElementById('detectRMSE').textContent = data.metrics.rmse.toFixed(4);
                    document.getElementById('detectCount').textContent = data.count;
                    document.getElementById('detectThreshold').textContent = data.threshold.toFixed(4);
                    
                    document.getElementById('detectContent').textContent = JSON.stringify(data, null, 2);
                    
                    // Draw chart
                    drawDetectChart(y_true, y_pred, data.anomalies);
                    chartEl.classList.add('show');
                })
                .catch(error => {
                    statusEl.textContent = '❌ Error: ' + error.message;
                    statusEl.className = 'status show error';
                });
            } catch (error) {
                statusEl.textContent = '❌ Error: ' + error.message;
                statusEl.className = 'status show error';
            }
        }

        function drawDetectChart(y_true, y_pred, anomalies) {
            const ctx = document.getElementById('detectCanvas').getContext('2d');
            
            if (detectChartInstance) {
                detectChartInstance.destroy();
            }

            const labels = Array.from({length: y_true.length}, (_, i) => i);
            const backgroundColor = labels.map(i => anomalies.includes(i) ? 'rgba(255, 99, 132, 0.5)' : 'rgba(54, 162, 235, 0.1)');

            detectChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Actual Values',
                            data: y_true,
                            borderColor: 'rgb(75, 192, 75)',
                            backgroundColor: 'rgba(75, 192, 75, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Predicted Values',
                            data: y_pred,
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Anomalies',
                            data: anomalies.map(idx => y_true[idx]),
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgb(255, 99, 132)',
                            showLine: false,
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            type: 'scatter'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Time Series with Predictions and Anomalies' }
                    },
                    scales: {
                        y: { beginAtZero: false }
                    }
                }
            });
        }

        function trainPredict() {
            const statusEl = document.getElementById('train-status');
            const chartEl = document.getElementById('trainChart');
            const btn = document.getElementById('trainBtn');
            
            try {
                const y_true = parseValues(document.getElementById('train_data').value);
                const sigma = parseFloat(document.getElementById('train_sigma').value);

                if (y_true.length < 10) {
                    throw new Error('Need at least 10 data points');
                }

                statusEl.textContent = '⏳ Training model... This may take a minute.';
                statusEl.className = 'status show info';
                btn.disabled = true;

                fetch('/train_predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ y_true, sigma })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) throw new Error(data.error);
                    
                    statusEl.textContent = '✅ Training Complete!';
                    statusEl.className = 'status show success';
                    btn.disabled = false;
                    
                    // Display metrics
                    document.getElementById('trainMetrics').style.display = 'grid';
                    document.getElementById('trainLayers').textContent = data.nas_config.layers;
                    document.getElementById('trainHidden').textContent = data.nas_config.hidden;
                    document.getElementById('trainMAE').textContent = data.metrics.mae.toFixed(4);
                    document.getElementById('trainRMSE').textContent = data.metrics.rmse.toFixed(4);
                    
                    document.getElementById('trainContent').textContent = JSON.stringify(data, null, 2);
                    
                    // Draw chart
                    drawTrainChart(y_true, data.y_pred, data.anomalies);
                    chartEl.classList.add('show');
                })
                .catch(error => {
                    statusEl.textContent = '❌ Error: ' + error.message;
                    statusEl.className = 'status show error';
                    btn.disabled = false;
                });
            } catch (error) {
                statusEl.textContent = '❌ Error: ' + error.message;
                statusEl.className = 'status show error';
            }
        }

        function drawTrainChart(y_true, y_pred, anomalies) {
            const ctx = document.getElementById('trainCanvas').getContext('2d');
            
            if (trainChartInstance) {
                trainChartInstance.destroy();
            }

            const labels = Array.from({length: y_true.length}, (_, i) => i);

            trainChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Actual Values',
                            data: y_true,
                            borderColor: 'rgb(75, 192, 75)',
                            backgroundColor: 'rgba(75, 192, 75, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Predicted Values (NAS-optimized)',
                            data: y_pred,
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            borderWidth: 2,
                            fill: false,
                            tension: 0.3
                        },
                        {
                            label: 'Detected Anomalies',
                            data: anomalies.map(idx => y_true[idx]),
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgb(255, 99, 132)',
                            showLine: false,
                            pointRadius: 6,
                            pointHoverRadius: 8,
                            type: 'scatter'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'NAS-Optimized Model Predictions with Anomalies' }
                    },
                    scales: {
                        y: { beginAtZero: false }
                    }
                }
            });
        }

        function loadSampleData() {
            const statusEl = document.getElementById('data-status');
            const chartEl = document.getElementById('dataChart');
            
            statusEl.textContent = '⏳ Loading...';
            statusEl.className = 'status show info';

            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    statusEl.textContent = '✅ Loaded!';
                    statusEl.className = 'status show success';
                    
                    document.getElementById('dataContent').textContent = JSON.stringify(data, null, 2);
                    
                    // Draw chart
                    drawDataChart(data.data);
                    chartEl.classList.add('show');
                })
                .catch(error => {
                    statusEl.textContent = '❌ Error: ' + error.message;
                    statusEl.className = 'status show error';
                });
        }

        function drawDataChart(data) {
            const ctx = document.getElementById('dataCanvas').getContext('2d');
            
            if (dataChartInstance) {
                dataChartInstance.destroy();
            }

            const labels = Array.from({length: data.length}, (_, i) => i);
            
            dataChartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Feature 1',
                            data: data.map(row => row[0]),
                            borderColor: 'rgb(75, 192, 75)',
                            backgroundColor: 'rgba(75, 192, 75, 0.1)',
                            borderWidth: 2,
                            tension: 0.3
                        },
                        {
                            label: 'Feature 2',
                            data: data.map(row => row[1]),
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            borderWidth: 2,
                            tension: 0.3
                        },
                        {
                            label: 'Feature 3',
                            data: data.map(row => row[2]),
                            borderColor: 'rgb(255, 159, 64)',
                            backgroundColor: 'rgba(255, 159, 64, 0.1)',
                            borderWidth: 2,
                            tension: 0.3
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { position: 'top' },
                        title: { display: true, text: 'Sample Dataset with Anomalies' }
                    },
                    scales: {
                        y: { beginAtZero: false }
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve HTML interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/data')
def data_endpoint():
    """Return generated time series data"""
    try:
        data = get_data()
        return jsonify({
            "data": data.tolist()[:100],  # First 100 samples
            "shape": [int(data.shape[0]), int(data.shape[1])],
            "features": 3,
            "note": "Returns first 100 samples for demo"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

from nas_search import nas_search
from train import train_and_eval

@app.route('/train_predict', methods=['POST'])
def train_predict():
    """
    1. Receives Actual Data
    2. Runs NAS to find best model
    3. Trains model
    4. Predicts values
    5. Detects Anomalies
    """
    try:
        req_data = request.json
        y_true = np.array(req_data.get('y_true', []), dtype=float)
        sigma = float(req_data.get('sigma', 3))
        
        if len(y_true) < 10:
            return jsonify({"error": "Need at least 10 data points to train"}), 400

        # Prepare data for LSTM (X=past, y=current)
        def create_sequences(data, seq_length=3):
            xs, ys = [], []
            for i in range(len(data)-seq_length):
                x = data[i:i+seq_length]
                y = data[i+seq_length]
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        SEQ_LEN = 3
        X, y = create_sequences(y_true, SEQ_LEN)
        
        # Reshape X for LSTM [samples, time steps, features]
        # Here we have 1 feature (univariate)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split for NAS (simple 80/20 split)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        print("🧠 Starting Neural Architecture Search (NAS)...")
        # Run NAS to find best config
        best_config, best_score = nas_search(train_and_eval, X_train, y_train, X_val, y_val)
        print(f"✅ Best config found: {best_config}")
        
        # Train final model with best config
        best_layers, best_hidden = best_config
        model, rmse = train_and_eval(best_layers, best_hidden, X_train, y_train, X_val, y_val, epochs=50)
        
        # Predict on entire dataset
        model.eval()
        with torch.no_grad():
            full_X_tensor = torch.FloatTensor(X).to(torch.device("cpu")) # CPU for interference
            if torch.cuda.is_available():
                model = model.cpu() # Move model to cpu for consistency
            
            predictions = model(full_X_tensor).numpy().flatten()
            
        # Pad predictions to match original length (first SEQ_LEN points have no prediction)
        # We assume the first SEQ_LEN predictions are just the actual values (perfect fit) or 0
        padded_pred = np.concatenate([y_true[:SEQ_LEN], predictions])
        
        # Detect Anomalies
        anomalies, threshold = detect_anomalies(y_true, padded_pred, sigma=sigma)
        metrics = evaluate_model(y_true, padded_pred)
        
        return jsonify({
            "y_pred": padded_pred.tolist(),
            "anomalies": anomalies.tolist(),
            "threshold": float(threshold),
            "nas_config": {"layers": best_layers, "hidden": best_hidden},
            "metrics": {
                "mae": float(metrics['mae']),
                "rmse": float(metrics['rmse'])
            }
        })

    except Exception as e:
        print(f"Error in training: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    """Detect anomalies given predictions
    
    GET: Returns instructions
    POST: Requires JSON with y_true, y_pred, and optional sigma
    
    Request JSON:
    {
        "y_true": [list of actual values],
        "y_pred": [list of predicted values],
        "sigma": 3 (optional, default 3)
    }
    """
    if request.method == 'GET':
        return jsonify({
            "message": "Use POST method to detect anomalies",
            "example": {
                "y_true": [1, 2, 3, 10, 5, 6],
                "y_pred": [1.1, 2.1, 2.9, 5, 5.2, 6.1],
                "sigma": 3
            },
            "description": "sigma parameter controls sensitivity (higher = stricter)"
        }), 200
    
    try:
        req_data = request.json
        if not req_data:
            return jsonify({"error": "Request body must be JSON"}), 400
            
        y_true = np.array(req_data.get('y_true', []), dtype=float)
        y_pred = np.array(req_data.get('y_pred', []), dtype=float)
        sigma = float(req_data.get('sigma', 3))
        
        if len(y_true) == 0 or len(y_pred) == 0:
            return jsonify({"error": "y_true and y_pred required"}), 400
        
        if len(y_true) != len(y_pred):
            return jsonify({"error": "y_true and y_pred must have same length"}), 400
        
        anomalies, threshold = detect_anomalies(y_true, y_pred, sigma=sigma)
        metrics = evaluate_model(y_true, y_pred)
        
        return jsonify({
            "anomalies": anomalies.tolist(),
            "threshold": float(threshold),
            "count": int(len(anomalies)),
            "metrics": {
                "mae": float(metrics['mae']),
                "rmse": float(metrics['rmse'])
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    print("Starting Time Series Anomaly Detection Server...")
    print("Visit http://localhost:5000 to get started")
    app.run(debug=False, host='127.0.0.1', port=5000)
