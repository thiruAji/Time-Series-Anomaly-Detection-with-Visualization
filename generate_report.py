# generate_report.py
"""
Generates comprehensive reports for Time Series Forecasting with Attention LSTM.
Includes NAS results, attention analysis, and baseline comparison.
"""
import os
from datetime import datetime
from typing import Dict, List
import numpy as np


def calculate_anomaly_metrics(predicted_anomalies, true_anomalies, n_total: int):
    """Calculate precision, recall, F1 for anomaly detection."""
    pred_set = set(predicted_anomalies.tolist() if hasattr(predicted_anomalies, 'tolist') else predicted_anomalies)
    true_set = set(true_anomalies.tolist() if hasattr(true_anomalies, 'tolist') else true_anomalies)
    
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn
    }


def generate_forecasting_report(
    search_history: List[Dict],
    best_config: Dict,
    best_score: float,
    baseline_comparison: Dict,
    attention_analysis: Dict,
    data_info: Dict,
    output_path: str = "results/forecasting_report.txt"
):
    """
    Generate comprehensive text report for time series forecasting.
    
    Args:
        search_history: NAS evolution history
        best_config: Best attention LSTM configuration
        best_score: Best validation RMSE
        baseline_comparison: Results comparing attention vs baseline
        attention_analysis: Attention weight analysis
        data_info: Dataset information
        output_path: Path to save report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Header
        f.write("=" * 70 + "\n")
        f.write("       TIME SERIES FORECASTING WITH ATTENTION LSTM\n")
        f.write("       Neural Architecture Search Results Report\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Dataset Information
        f.write("-" * 70 + "\n")
        f.write("1. DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Total Samples:     {data_info.get('n_samples', 'N/A')}\n")
        f.write(f"   Features:          {data_info.get('n_features', 'N/A')}\n")
        f.write(f"   Sequence Length:   {data_info.get('seq_length', 'N/A')}\n")
        f.write(f"   Forecast Horizon:  {data_info.get('forecast_horizon', 1)}\n")
        f.write(f"   Train/Val Split:   {data_info.get('train_ratio', 0.8)*100:.0f}% / "
                f"{(1-data_info.get('train_ratio', 0.8))*100:.0f}%\n\n")
        
        # 2. NAS Search Space
        f.write("-" * 70 + "\n")
        f.write("2. NAS SEARCH SPACE\n")
        f.write("-" * 70 + "\n")
        f.write("   Parameter           Options\n")
        f.write("   ----------------    ----------------------------------\n")
        f.write("   Attention Type      Bahdanau (Additive), Luong (Multiplicative)\n")
        f.write("   Attention Dim       32, 64, 128\n")
        f.write("   Num Layers          1, 2, 3\n")
        f.write("   Hidden Size         32, 64, 128\n")
        f.write("   Dropout             0.0, 0.1, 0.2, 0.3\n")
        f.write("   Learning Rate       0.001, 0.005, 0.01\n")
        total_configs = 2 * 3 * 3 * 3 * 4 * 3
        f.write(f"   Total Configs:      {total_configs} possible combinations\n\n")
        
        # 3. Evolutionary Search Results
        f.write("-" * 70 + "\n")
        f.write("3. EVOLUTIONARY SEARCH RESULTS\n")
        f.write("-" * 70 + "\n")
        if search_history:
            f.write(f"   Generations:        {len(search_history)}\n")
            f.write(f"   Population Size:    {len(search_history[0].get('population_scores', []))}\n\n")
            
            f.write("   Generation Progress:\n")
            for gen in search_history:
                scores = gen.get('population_scores', [])
                valid_scores = [s for s in scores if s < float('inf')]
                if valid_scores:
                    f.write(f"     Gen {gen['generation']}: Best={min(valid_scores):.4f}, "
                            f"Avg={np.mean(valid_scores):.4f}, Worst={max(valid_scores):.4f}\n")
        f.write("\n")
        
        # 4. Best Architecture
        f.write("-" * 70 + "\n")
        f.write("4. BEST ATTENTION LSTM ARCHITECTURE\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Attention Type:    {best_config.get('attention_type', 'bahdanau').capitalize()}\n")
        f.write(f"   Attention Dim:     {best_config.get('attention_dim', 64)}\n")
        f.write(f"   Num Layers:        {best_config.get('num_layers', 2)}\n")
        f.write(f"   Hidden Size:       {best_config.get('hidden_size', 64)}\n")
        f.write(f"   Dropout:           {best_config.get('dropout', 0.2)}\n")
        f.write(f"   Learning Rate:     {best_config.get('learning_rate', 0.001)}\n")
        f.write(f"   Validation RMSE:   {best_score:.6f}\n\n")
        
        # 5. Baseline Comparison
        f.write("-" * 70 + "\n")
        f.write("5. ATTENTION VS BASELINE COMPARISON\n")
        f.write("-" * 70 + "\n")
        if baseline_comparison:
            f.write(f"   Baseline LSTM RMSE:    {baseline_comparison.get('baseline_rmse', 'N/A'):.6f}\n")
            f.write(f"   Attention LSTM RMSE:   {baseline_comparison.get('attention_rmse', 'N/A'):.6f}\n")
            improvement = baseline_comparison.get('improvement_percent', 0)
            if improvement > 0:
                f.write(f"   Improvement:           {improvement:.2f}% better with attention\n")
            else:
                f.write(f"   Difference:            {abs(improvement):.2f}%\n")
        else:
            f.write("   Baseline comparison not performed.\n")
        f.write("\n")
        
        # 6. Attention Analysis
        f.write("-" * 70 + "\n")
        f.write("6. ATTENTION WEIGHT ANALYSIS\n")
        f.write("-" * 70 + "\n")
        if attention_analysis:
            f.write(f"   Peak Attention Step:   {attention_analysis.get('peak_attention_step', 'N/A')}\n")
            f.write(f"   Peak Value:            {attention_analysis.get('peak_attention_value', 0):.4f}\n")
            f.write(f"   Mean Entropy:          {attention_analysis.get('mean_entropy', 0):.4f}\n")
            f.write(f"   Recent Steps Focus:    {attention_analysis.get('recent_attention_ratio', 0):.4f}\n")
            f.write(f"   Early Steps Focus:     {attention_analysis.get('early_attention_ratio', 0):.4f}\n\n")
            
            interpretation = attention_analysis.get('interpretation', '')
            if interpretation:
                f.write("   Interpretation:\n")
                words = interpretation.split()
                line = "   "
                for word in words:
                    if len(line) + len(word) > 68:
                        f.write(line + "\n")
                        line = "   " + word + " "
                    else:
                        line += word + " "
                f.write(line + "\n")
        else:
            f.write("   Attention analysis not performed.\n")
        f.write("\n")
        
        # Footer
        f.write("=" * 70 + "\n")
        f.write("                      END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Report saved to: {output_path}")
    return output_path


# Backward compatibility
def generate_nas_report(search_history, best_config, best_score, anomaly_metrics, data_info, output_path="results/nas_report.txt"):
    """Legacy wrapper for report generation."""
    return generate_forecasting_report(
        search_history=search_history,
        best_config=best_config,
        best_score=best_score,
        baseline_comparison=None,
        attention_analysis=None,
        data_info=data_info,
        output_path=output_path
    )


if __name__ == "__main__":
    # Test report generation
    test_history = [
        {'generation': 1, 'best_score': 0.5, 'population_scores': [0.5, 0.6, 0.7]},
        {'generation': 2, 'best_score': 0.4, 'population_scores': [0.4, 0.5, 0.55]},
    ]
    test_config = {
        'num_layers': 2, 
        'hidden_size': 128, 
        'attention_type': 'bahdanau',
        'attention_dim': 64,
        'dropout': 0.1, 
        'learning_rate': 0.005
    }
    test_baseline = {'baseline_rmse': 0.45, 'attention_rmse': 0.35, 'improvement_percent': 22.2}
    test_attention = {
        'peak_attention_step': 6,
        'peak_attention_value': 0.35,
        'mean_entropy': 0.65,
        'recent_attention_ratio': 0.70,
        'early_attention_ratio': 0.18,
        'interpretation': "The model focuses primarily on the most recent time steps."
    }
    test_data = {'n_samples': 1000, 'n_features': 5, 'seq_length': 10, 'train_ratio': 0.8}
    
    generate_forecasting_report(test_history, test_config, 0.35, test_baseline, test_attention, test_data)
