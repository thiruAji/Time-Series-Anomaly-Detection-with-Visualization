# 📈 Time Series Anomaly Detection with Neural Architecture Search (NAS)

An advanced ML application for detecting anomalies in time-series data using **Evolutionary Neural Architecture Search** to automatically discover optimal RNN architectures.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)

## 🚀 Key Features

### Neural Architecture Search (Evolutionary Strategy)
- **Population-based search**: Evolves a population of model configurations over multiple generations
- **Cell type exploration**: Searches over both **LSTM** and **GRU** architectures
- **Hyperparameter tuning**: Optimizes layers (1-3), hidden units (32-256), dropout (0-0.3), learning rate
- **Selection + Mutation + Crossover**: Elite selection, random mutation, parent crossover

### Complex Data Generation
- **5-feature multivariate data**: base, trend, seasonality, momentum, volatility
- **Injected anomalies**: Ground truth anomalies for rigorous evaluation
- **Configurable**: Adjustable steps, features, and anomaly ratio

### Rigorous Evaluation
- **Precision/Recall/F1**: Evaluates detection against known anomalies
- **Text Reports**: Generates comprehensive `nas_report.txt` with full search history

## 📂 Project Structure

```
├── app.py                 # Flask API with /train_predict endpoint
├── server.py              # Production Waitress server
├── nas_search.py          # Evolutionary NAS implementation
├── model.py               # RNNModel (LSTM/GRU) definition
├── train.py               # Training & evaluation module
├── data_generation.py     # 5-feature synthetic data generator
├── generate_report.py     # Text-based NAS report generator
├── anomaly_detection.py   # Residual-based anomaly detection
├── evaluation.py          # MAE/RMSE metrics
├── results/               # Generated reports directory
│   └── nas_report.txt     # Latest NAS search report
└── templates/
    └── index.html         # Web interface
```

## 🛠️ Installation

```bash
git clone https://github.com/thiruAji/Time-Series-Anomaly-Detection-with-Visualization.git
cd Time-Series-Anomaly-Detection-with-Visualization
pip install -r requirements.txt
```

## 🏃 Usage

### Start Server
```bash
python server.py
```
Visit `http://localhost:5001`

### API: Run NAS Pipeline
```python
import requests

# With user data
response = requests.post('http://localhost:5001/train_predict', json={
    'y_true': [1, 2, 3, 10, 5, 6, 7, 8, 9, 10, 5, 4, 3, 2, 1],
    'sigma': 3
})

# With synthetic 5-feature data
response = requests.post('http://localhost:5001/train_predict', json={
    'use_synthetic': True,
    'sigma': 3
})

print(response.json())
```

### Response Structure
```json
{
  "y_pred": [...],
  "anomalies": [3, 10],
  "count": 2,
  "threshold": 0.8543,
  "nas_config": {
    "cell_type": "GRU",
    "layers": 2,
    "hidden": 128,
    "dropout": 0.1,
    "learning_rate": 0.005
  },
  "metrics": {"mae": 0.234, "rmse": 0.312, "val_rmse": 0.298},
  "anomaly_eval": {"precision": 0.67, "recall": 0.80, "f1_score": 0.73},
  "report_path": "results/nas_report.txt",
  "search_summary": {"generations": 4, "configs_evaluated": 32}
}
```

## 🧬 NAS Search Space

| Parameter | Options |
|-----------|---------|
| Cell Type | LSTM, GRU |
| Num Layers | 1, 2, 3 |
| Hidden Size | 32, 64, 128, 256 |
| Dropout | 0.0, 0.1, 0.2, 0.3 |
| Learning Rate | 0.001, 0.005, 0.01 |

**Total Configurations**: 288 possible architectures

## 📊 Algorithmic Details

1. **Data Preparation**: Input is transformed into 5 engineered features
2. **Sequence Creation**: Sliding window creates (X, y) pairs for RNN training
3. **Evolutionary NAS**: 
   - Initialize random population
   - Train each config, evaluate RMSE
   - Select top performers (elites)
   - Create next generation via mutation/crossover
   - Repeat for N generations
4. **Prediction**: Best model predicts on full dataset
5. **Anomaly Detection**: Residual analysis with configurable sigma threshold
6. **Evaluation**: Compare detected anomalies against ground truth (if available)
7. **Reporting**: Generate text-based report with all details

## 📄 License

MIT License - see [LICENSE](LICENSE) file.