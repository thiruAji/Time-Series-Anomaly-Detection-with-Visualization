# model.py
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        # Return predictions for last timestep
        # out shape: (batch_size, seq_len, hidden_size)
        # We want the last time step: out[:, -1, :]
        if out.dim() == 2:
             # Handle case where batch_first=True might return squeezed output for single layer? 
             # No, standard LSTM output is 3D with batch_first=True
             pass
        return self.fc(out[:, -1, :])
