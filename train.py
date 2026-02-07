# train.py
"""
Training module for Attention-based LSTM with walk-forward cross-validation.
Supports both attention and baseline models for comparison.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import AttentionLSTM, BaselineLSTM


def walk_forward_split(X, y, n_splits=5, test_size=0.2):
    """
    Generate walk-forward cross-validation splits for time series.
    
    Unlike standard k-fold, this ensures training data always precedes test data,
    which is critical for time series to prevent data leakage.
    
    Args:
        X, y: Full dataset
        n_splits: Number of validation folds
        test_size: Fraction of data to use as test in each split
    
    Yields:
        (X_train, y_train, X_val, y_val) for each split
    """
    n_samples = len(X)
    test_samples = int(n_samples * test_size)
    
    for i in range(n_splits):
        # Calculate split points
        # Each fold uses progressively more training data
        train_end = int(n_samples * (0.5 + i * 0.1))  # Start at 50%, grow by 10% each fold
        train_end = min(train_end, n_samples - test_samples)
        
        val_start = train_end
        val_end = min(val_start + test_samples, n_samples)
        
        if val_start >= val_end or train_end <= 0:
            continue
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]
        
        yield X_train, y_train, X_val, y_val


def train_model(model, X_train, y_train, epochs=30, batch_size=32, lr=0.001):
    """
    Train a model on given data.
    
    Args:
        model: PyTorch model
        X_train, y_train: Training data
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        Trained model, list of epoch losses
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Handle shape mismatch
            if outputs.shape != y_batch.shape:
                if len(y_batch.shape) == 1:
                    outputs = outputs.squeeze()
                elif outputs.shape[-1] != y_batch.shape[-1]:
                    outputs = outputs[:, :y_batch.shape[-1]]
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(train_loader))
    
    return model, losses


def evaluate_model(model, X_val, y_val):
    """
    Evaluate model on validation data.
    
    Returns:
        dict with rmse, mae, predictions
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        predictions = model(X_val_t)
        
        # Handle shape
        if predictions.shape != y_val_t.shape:
            if len(y_val_t.shape) == 1:
                predictions = predictions.squeeze()
        
        mse = nn.MSELoss()(predictions, y_val_t).item()
        rmse = np.sqrt(mse)
        mae = torch.mean(torch.abs(predictions - y_val_t)).item()
    
    return {
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions.cpu().numpy()
    }


def train_and_eval(config, X_train, y_train, X_val, y_val, epochs=30):
    """
    Train and evaluate model with given configuration.
    
    Args:
        config: Dict with model configuration OR legacy tuple (layers, hidden)
        X_train, y_train, X_val, y_val: Data splits
        epochs: Training epochs
    
    Returns:
        (model, rmse)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle legacy format
    if isinstance(config, tuple):
        layers, hidden = config
        config = {
            'num_layers': layers,
            'hidden_size': hidden,
            'attention_type': 'bahdanau',
            'attention_dim': 64,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'use_attention': True
        }
    
    # Determine input size
    input_size = X_train.shape[2] if len(X_train.shape) == 3 else 1
    
    # Build model (attention or baseline)
    if config.get('use_attention', True):
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            attention_type=config.get('attention_type', 'bahdanau'),
            attention_dim=config.get('attention_dim', 64),
            dropout=config.get('dropout', 0.2)
        ).to(device)
    else:
        model = BaselineLSTM(
            input_size=input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config.get('dropout', 0.2)
        ).to(device)
    
    # Train
    model, losses = train_model(
        model, X_train, y_train,
        epochs=epochs,
        lr=config.get('learning_rate', 0.001)
    )
    
    # Evaluate
    metrics = evaluate_model(model, X_val, y_val)
    
    return model, metrics['rmse']


def compare_attention_vs_baseline(X_train, y_train, X_val, y_val, hidden_size=64, num_layers=2):
    """
    Compare Attention LSTM against Baseline LSTM.
    
    Returns:
        dict with comparison results
    """
    input_size = X_train.shape[2]
    
    # Train Baseline
    print("Training Baseline LSTM (no attention)...")
    baseline_config = {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'use_attention': False,
        'dropout': 0.2,
        'learning_rate': 0.001
    }
    baseline_model, baseline_rmse = train_and_eval(baseline_config, X_train, y_train, X_val, y_val)
    
    # Train Attention LSTM
    print("Training Attention LSTM (Bahdanau)...")
    attention_config = {
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'use_attention': True,
        'attention_type': 'bahdanau',
        'attention_dim': 64,
        'dropout': 0.2,
        'learning_rate': 0.001
    }
    attention_model, attention_rmse = train_and_eval(attention_config, X_train, y_train, X_val, y_val)
    
    improvement = ((baseline_rmse - attention_rmse) / baseline_rmse) * 100
    
    return {
        'baseline_rmse': baseline_rmse,
        'attention_rmse': attention_rmse,
        'improvement_percent': improvement,
        'attention_model': attention_model,
        'baseline_model': baseline_model
    }


if __name__ == "__main__":
    from data_generation import generate_data
    
    # Generate test data
    data, anomalies = generate_data(n_steps=300)
    X = data.values[:-1].reshape(-1, 1, 5)
    y = data.values[1:]
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # Compare models
    results = compare_attention_vs_baseline(X_train, y_train, X_val, y_val)
    
    print(f"\nResults:")
    print(f"  Baseline LSTM RMSE: {results['baseline_rmse']:.4f}")
    print(f"  Attention LSTM RMSE: {results['attention_rmse']:.4f}")
    print(f"  Improvement: {results['improvement_percent']:.2f}%")
