# model.py
"""
Flexible RNN Model supporting both LSTM and GRU cells.
"""
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """
    Configurable RNN model for time series forecasting.
    Supports LSTM and GRU cell types for NAS search space.
    """
    def __init__(self, input_size, hidden_size, num_layers, cell_type='LSTM', dropout=0.2):
        """
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units per layer
            num_layers: Number of stacked RNN layers
            cell_type: 'LSTM' or 'GRU'
            dropout: Dropout rate (applied if num_layers > 1)
        """
        super().__init__()
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Select RNN cell type
        rnn_class = nn.LSTM if cell_type == 'LSTM' else nn.GRU
        
        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
        
        Returns:
            Output tensor of shape (batch, features)
        """
        # RNN forward
        out, _ = self.rnn(x)
        
        # Take the last timestep output
        # out shape: (batch, seq_len, hidden_size)
        last_output = out[:, -1, :]
        
        # Project to output size
        return self.fc(last_output)


# Backward compatibility alias
LSTMModel = RNNModel


if __name__ == "__main__":
    # Test both model types
    batch, seq, features = 32, 10, 5
    x = torch.randn(batch, seq, features)
    
    for cell_type in ['LSTM', 'GRU']:
        model = RNNModel(input_size=features, hidden_size=64, num_layers=2, cell_type=cell_type)
        out = model(x)
        print(f"{cell_type} output shape: {out.shape}")
