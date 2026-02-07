# generate_report.py
"""
Generates text-based NAS report with search results, architecture, and anomaly detection metrics.
"""
import os
import json
from datetime import datetime
from typing import Dict, List
import numpy as np

def calculate_anomaly_metrics(predicted_anomalies: np.ndarray, true_anomalies: np.ndarray, n_total: int):
    """
    Calculate precision, recall, F1 for anomaly detection.
    
    Args:
        predicted_anomalies: Array of predicted anomaly indices
        true_anomalies: Array of ground truth anomaly indices
        n_total: Total number of data points
    
    Returns:
        Dict with precision, recall, f1, tp, fp, fn
    """
    pred_set = set(predicted_anomalies.tolist() if hasattr(predicted_anomalies, 'tolist') else predicted_anomalies)
    true_set = set(true_anomalies.tolist() if hasattr(true_anomalies, 'tolist') else true_anomalies)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = n_total - tp - fp - fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'true_negatives': tn
    }


def generate_nas_report(
    search_history: List[Dict],
    best_config: Dict,
    best_score: float,
    anomaly_metrics: Dict,
    data_info: Dict,
    output_path: str = "results/nas_report.txt"
):
    """
    Generate a comprehensive text report of NAS results.
    
    Args:
        search_history: List of generation results from evolutionary search
        best_config: Best architecture configuration found
        best_score: Best validation RMSE
        anomaly_metrics: Precision/recall/F1 metrics
        data_info: Information about the dataset
        output_path: Path to save the report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("       NEURAL ARCHITECTURE SEARCH (NAS) REPORT\n")
        f.write("       Time Series Anomaly Detection System\n")
        f.write("=" * 70 + "\n\n")
        
        # Timestamp
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Data Information
        f.write("-" * 70 + "\n")
        f.write("1. DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Total Samples:     {data_info.get('n_samples', 'N/A')}\n")
        f.write(f"   Features:          {data_info.get('n_features', 'N/A')}\n")
        f.write(f"   Feature Names:     {data_info.get('feature_names', 'N/A')}\n")
        f.write(f"   Injected Anomalies:{data_info.get('n_anomalies', 'N/A')}\n")
        f.write(f"   Train/Val Split:   {data_info.get('train_ratio', 0.8)*100:.0f}% / {(1-data_info.get('train_ratio', 0.8))*100:.0f}%\n\n")
        
        # Search Space
        f.write("-" * 70 + "\n")
        f.write("2. NAS SEARCH SPACE\n")
        f.write("-" * 70 + "\n")
        f.write("   Parameter         Options\n")
        f.write("   ---------------   ----------------------------------\n")
        f.write("   Cell Type         LSTM, GRU\n")
        f.write("   Num Layers        1, 2, 3\n")
        f.write("   Hidden Size       32, 64, 128, 256\n")
        f.write("   Dropout           0.0, 0.1, 0.2, 0.3\n")
        f.write("   Learning Rate     0.001, 0.005, 0.01\n")
        f.write(f"   Total Configs:    {2 * 3 * 4 * 4 * 3} possible combinations\n\n")
        
        # Evolutionary Search Summary
        f.write("-" * 70 + "\n")
        f.write("3. EVOLUTIONARY SEARCH RESULTS\n")
        f.write("-" * 70 + "\n")
        if search_history:
            f.write(f"   Generations:       {len(search_history)}\n")
            f.write(f"   Population Size:   {len(search_history[0].get('population_scores', []))}\n\n")
            
            f.write("   Generation Progress:\n")
            for gen in search_history:
                scores = gen.get('population_scores', [])
                valid_scores = [s for s in scores if s < float('inf')]
                if valid_scores:
                    f.write(f"     Gen {gen['generation']}: Best={min(valid_scores):.4f}, Avg={np.mean(valid_scores):.4f}, Worst={max(valid_scores):.4f}\n")
        f.write("\n")
        
        # Best Architecture
        f.write("-" * 70 + "\n")
        f.write("4. BEST ARCHITECTURE FOUND\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Cell Type:         {best_config.get('cell_type', 'LSTM')}\n")
        f.write(f"   Num Layers:        {best_config.get('num_layers', 'N/A')}\n")
        f.write(f"   Hidden Size:       {best_config.get('hidden_size', 'N/A')}\n")
        f.write(f"   Dropout:           {best_config.get('dropout', 'N/A')}\n")
        f.write(f"   Learning Rate:     {best_config.get('learning_rate', 'N/A')}\n")
        f.write(f"   Validation RMSE:   {best_score:.6f}\n\n")
        
        # Anomaly Detection Performance
        f.write("-" * 70 + "\n")
        f.write("5. ANOMALY DETECTION PERFORMANCE\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Precision:         {anomaly_metrics.get('precision', 0):.4f}\n")
        f.write(f"   Recall:            {anomaly_metrics.get('recall', 0):.4f}\n")
        f.write(f"   F1 Score:          {anomaly_metrics.get('f1_score', 0):.4f}\n")
        f.write(f"   True Positives:    {anomaly_metrics.get('true_positives', 0)}\n")
        f.write(f"   False Positives:   {anomaly_metrics.get('false_positives', 0)}\n")
        f.write(f"   False Negatives:   {anomaly_metrics.get('false_negatives', 0)}\n\n")
        
        # Footer
        f.write("=" * 70 + "\n")
        f.write("                      END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test report generation
    test_history = [
        {'generation': 1, 'best_score': 0.5, 'best_config': {}, 'population_scores': [0.5, 0.6, 0.7]},
        {'generation': 2, 'best_score': 0.4, 'best_config': {}, 'population_scores': [0.4, 0.5, 0.55]},
    ]
    test_config = {'num_layers': 2, 'hidden_size': 128, 'cell_type': 'GRU', 'dropout': 0.1, 'learning_rate': 0.005}
    test_metrics = calculate_anomaly_metrics([5, 10, 15, 20], [5, 10, 25], 100)
    test_data_info = {'n_samples': 1000, 'n_features': 5, 'feature_names': ['base', 'trend', 'seasonality', 'momentum', 'volatility'], 'n_anomalies': 30}
    
    generate_nas_report(test_history, test_config, 0.35, test_metrics, test_data_info)
