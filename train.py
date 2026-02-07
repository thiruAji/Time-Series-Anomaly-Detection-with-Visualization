# train.py
"""
Training module for RNN models with configurable hyperparameters.
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import RNNModel

def train_and_eval(config, X_train, y_train, X_val, y_val, epochs=30):
    """
    Train and evaluate an RNN model with given configuration.
    
    Args:
        config: Dict with keys: num_layers, hidden_size, cell_type, dropout, learning_rate
               OR legacy format: (layers, hidden) tuple
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Number of training epochs
    
    Returns:
        Tuple of (trained_model, validation_rmse)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle legacy format
    if isinstance(config, tuple):
        layers, hidden = config
        config = {
            'num_layers': layers,
            'hidden_size': hidden,
            'cell_type': 'LSTM',
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    
    # Determine input size from data shape (samples, seq_len, features)
    input_size = X_train.shape[2] if len(X_train.shape) == 3 else 1
    
    # Build model
    model = RNNModel(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        cell_type=config.get('cell_type', 'LSTM'),
        dropout=config.get('dropout', 0.2)
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.get('learning_rate', 0.001)
    )
    
    # Prepare data loaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Handle output shape
            if outputs.shape != y_batch.shape:
                if len(y_batch.shape) == 1:
                    outputs = outputs.squeeze()
                elif outputs.shape[-1] != y_batch.shape[-1]:
                    outputs = outputs[:, :y_batch.shape[-1]]
            
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_val_t = torch.FloatTensor(X_val).to(device)
        y_val_t = torch.FloatTensor(y_val).to(device)
        
        val_outputs = model(X_val_t)
        
        # Handle output shape
        if val_outputs.shape != y_val_t.shape:
            if len(y_val_t.shape) == 1:
                val_outputs = val_outputs.squeeze()
            elif val_outputs.shape[-1] != y_val_t.shape[-1]:
                val_outputs = val_outputs[:, :y_val_t.shape[-1]]
        
        val_loss = criterion(val_outputs, y_val_t)
        rmse = np.sqrt(val_loss.cpu().item())
    
    return model, rmse


if __name__ == "__main__":
    # Quick test
    from data_generation import generate_data
    
    data, anomalies = generate_data(n_steps=200)
    X = data.values[:-1].reshape(-1, 1, 5)
    y = data.values[1:]
    
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    config = {
        'num_layers': 2,
        'hidden_size': 64,
        'cell_type': 'GRU',
        'dropout': 0.1,
        'learning_rate': 0.005
    }
    
    model, rmse = train_and_eval(config, X_train, y_train, X_val, y_val)
    print(f"Test RMSE: {rmse:.4f}")
