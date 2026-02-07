# attention_analysis.py
"""
Attention Weight Analysis and Visualization for Time Series Forecasting.
Provides tools to interpret which time steps the model focuses on.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import os


def analyze_attention_weights(attention_weights, save_path="results/attention_analysis.png"):
    """
    Analyze and visualize attention weight patterns.
    
    Args:
        attention_weights: numpy array of shape (n_samples, seq_len)
        save_path: Path to save visualization
    
    Returns:
        dict with analysis results
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    n_samples, seq_len = attention_weights.shape
    
    # Compute statistics
    mean_weights = attention_weights.mean(axis=0)
    std_weights = attention_weights.std(axis=0)
    
    # Find which time steps get most attention on average
    peak_attention_step = np.argmax(mean_weights)
    peak_attention_value = mean_weights[peak_attention_step]
    
    # Analyze temporal patterns
    # Check if attention is concentrated on recent steps (expected for forecasting)
    recent_attention = mean_weights[-3:].sum()  # Last 3 steps
    early_attention = mean_weights[:3].sum()    # First 3 steps
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Mean attention weights across time steps
    ax1 = axes[0, 0]
    steps = np.arange(seq_len)
    ax1.bar(steps, mean_weights, color='steelblue', alpha=0.7)
    ax1.fill_between(steps, mean_weights - std_weights, mean_weights + std_weights, 
                     alpha=0.3, color='steelblue')
    ax1.set_xlabel('Time Step (t-n to t-1)')
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('Mean Attention Distribution Across Input Sequence')
    ax1.axvline(x=peak_attention_step, color='red', linestyle='--', 
                label=f'Peak at step {peak_attention_step}')
    ax1.legend()
    
    # Plot 2: Heatmap of attention weights for sample predictions
    ax2 = axes[0, 1]
    n_show = min(50, n_samples)
    im = ax2.imshow(attention_weights[:n_show], aspect='auto', cmap='viridis')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Sample Index')
    ax2.set_title(f'Attention Heatmap (First {n_show} Samples)')
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: Attention entropy (diversity of focus)
    ax3 = axes[1, 0]
    entropy = -np.sum(attention_weights * np.log(attention_weights + 1e-10), axis=1)
    max_entropy = np.log(seq_len)
    normalized_entropy = entropy / max_entropy
    ax3.hist(normalized_entropy, bins=30, color='coral', alpha=0.7)
    ax3.axvline(x=normalized_entropy.mean(), color='red', linestyle='--',
                label=f'Mean: {normalized_entropy.mean():.3f}')
    ax3.set_xlabel('Normalized Entropy')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Attention Focus Distribution (0=focused, 1=uniform)')
    ax3.legend()
    
    # Plot 4: Cumulative attention by recency
    ax4 = axes[1, 1]
    cumulative = np.cumsum(mean_weights[::-1])[::-1]  # Reverse cumsum
    ax4.plot(steps, cumulative, 'o-', color='green')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Cumulative Attention (from step to end)')
    ax4.set_title('Cumulative Attention by Recency')
    ax4.axhline(y=0.5, color='red', linestyle='--', label='50% attention threshold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    # Prepare analysis report
    analysis = {
        'n_samples': n_samples,
        'sequence_length': seq_len,
        'peak_attention_step': int(peak_attention_step),
        'peak_attention_value': float(peak_attention_value),
        'mean_attention_weights': mean_weights.tolist(),
        'std_attention_weights': std_weights.tolist(),
        'recent_attention_ratio': float(recent_attention),
        'early_attention_ratio': float(early_attention),
        'mean_entropy': float(normalized_entropy.mean()),
        'visualization_path': save_path,
        'interpretation': generate_interpretation(mean_weights, peak_attention_step, 
                                                   normalized_entropy.mean(), 
                                                   recent_attention, early_attention)
    }
    
    return analysis


def generate_interpretation(mean_weights, peak_step, mean_entropy, recent, early):
    """Generate human-readable interpretation of attention patterns."""
    seq_len = len(mean_weights)
    
    lines = []
    
    # Temporal focus
    if peak_step >= seq_len - 2:
        lines.append("The model focuses primarily on the most recent time steps, "
                    "which is typical for short-term forecasting tasks.")
    elif peak_step <= 2:
        lines.append("The model shows unusual focus on early time steps, "
                    "suggesting it may be learning long-range dependencies or patterns.")
    else:
        lines.append(f"The model shows peak attention at step {peak_step}, "
                    "indicating it balances recent and historical information.")
    
    # Attention diversity
    if mean_entropy > 0.8:
        lines.append("Attention is relatively uniform across time steps, "
                    "suggesting all historical data contributes to predictions.")
    elif mean_entropy < 0.4:
        lines.append("Attention is highly concentrated on specific time steps, "
                    "indicating the model identifies key predictive moments.")
    else:
        lines.append("Attention shows moderate concentration, "
                    "balancing focus and diversity in information usage.")
    
    # Recent vs early
    if recent > early * 2:
        lines.append("Recent time steps receive significantly more attention than earlier ones, "
                    "consistent with typical time series forecasting behavior.")
    elif early > recent:
        lines.append("Earlier time steps receive more attention than recent ones, "
                    "which may indicate periodic patterns or delayed effects in the data.")
    
    return " ".join(lines)


def generate_attention_report(analysis, output_path="results/attention_report.txt"):
    """Generate text report of attention analysis."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("           ATTENTION WEIGHT ANALYSIS REPORT\n")
        f.write("           Time Series Forecasting with Attention LSTM\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("1. DATASET INFORMATION\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Samples Analyzed:    {analysis['n_samples']}\n")
        f.write(f"   Sequence Length:     {analysis['sequence_length']}\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("2. ATTENTION STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"   Peak Attention Step: {analysis['peak_attention_step']} "
                f"(weight: {analysis['peak_attention_value']:.4f})\n")
        f.write(f"   Recent Steps Focus:  {analysis['recent_attention_ratio']:.4f}\n")
        f.write(f"   Early Steps Focus:   {analysis['early_attention_ratio']:.4f}\n")
        f.write(f"   Mean Entropy:        {analysis['mean_entropy']:.4f} "
                f"(0=focused, 1=uniform)\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("3. INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        # Wrap text for readability
        interpretation = analysis['interpretation']
        words = interpretation.split()
        line = "   "
        for word in words:
            if len(line) + len(word) > 68:
                f.write(line + "\n")
                line = "   " + word + " "
            else:
                line += word + " "
        f.write(line + "\n\n")
        
        f.write("-" * 70 + "\n")
        f.write("4. MEAN ATTENTION WEIGHTS BY TIME STEP\n")
        f.write("-" * 70 + "\n")
        weights = analysis['mean_attention_weights']
        for i, w in enumerate(weights):
            bar = "#" * int(w * 50)
            f.write(f"   Step {i:2d}: {w:.4f} |{bar}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("                      END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Attention report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Test with synthetic attention weights
    np.random.seed(42)
    # Simulate attention that focuses on recent steps (realistic pattern)
    base_weights = np.array([0.05, 0.05, 0.08, 0.12, 0.15, 0.20, 0.35])
    # Add variation for each sample
    attention_weights = base_weights + np.random.normal(0, 0.02, (100, 7))
    attention_weights = np.abs(attention_weights)
    attention_weights = attention_weights / attention_weights.sum(axis=1, keepdims=True)
    
    analysis = analyze_attention_weights(attention_weights)
    generate_attention_report(analysis)
    
    print(f"\nAnalysis complete!")
    print(f"Peak attention at step: {analysis['peak_attention_step']}")
    print(f"Interpretation: {analysis['interpretation'][:200]}...")
