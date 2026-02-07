# model.py
"""
Attention-based LSTM/GRU for Time Series Forecasting.
Implements Bahdanau (Additive) Attention mechanism.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    """
    Bahdanau (Additive) Attention Mechanism.
    
    Computes attention weights over encoder hidden states.
    """
    def __init__(self, hidden_size, attention_dim=None):
        super().__init__()
        attention_dim = attention_dim or hidden_size
        
        # Attention layers
        self.W_h = nn.Linear(hidden_size, attention_dim, bias=False)  # For encoder hidden states
        self.W_s = nn.Linear(hidden_size, attention_dim, bias=False)  # For decoder state
        self.v = nn.Linear(attention_dim, 1, bias=False)              # Score vector
        
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: (batch, seq_len, hidden_size) - All encoder hidden states
            decoder_hidden: (batch, hidden_size) - Current decoder hidden state
        
        Returns:
            context: (batch, hidden_size) - Weighted sum of encoder outputs
            attention_weights: (batch, seq_len) - Attention distribution
        """
        seq_len = encoder_outputs.size(1)
        
        # Expand decoder hidden to match sequence length
        decoder_hidden_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Compute attention scores: tanh(W_h * h + W_s * s)
        energy = torch.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_hidden_expanded))
        
        # Compute attention weights
        scores = self.v(energy).squeeze(-1)  # (batch, seq_len)
        attention_weights = F.softmax(scores, dim=-1)
        
        # Compute context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class LuongAttention(nn.Module):
    """
    Luong (Multiplicative/Dot-Product) Attention Mechanism.
    """
    def __init__(self, hidden_size, method='general'):
        super().__init__()
        self.method = method
        
        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, encoder_outputs, decoder_hidden):
        """
        Args:
            encoder_outputs: (batch, seq_len, hidden_size)
            decoder_hidden: (batch, hidden_size)
        
        Returns:
            context, attention_weights
        """
        if self.method == 'dot':
            scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(-1)).squeeze(-1)
        elif self.method == 'general':
            scores = torch.bmm(self.W(encoder_outputs), decoder_hidden.unsqueeze(-1)).squeeze(-1)
        elif self.method == 'concat':
            seq_len = encoder_outputs.size(1)
            decoder_expanded = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            concat = torch.cat([encoder_outputs, decoder_expanded], dim=-1)
            scores = self.v(torch.tanh(self.W(concat))).squeeze(-1)
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context, attention_weights


class AttentionLSTM(nn.Module):
    """
    LSTM with Attention mechanism for Time Series Forecasting.
    
    Architecture:
        Input -> LSTM Encoder -> Attention -> FC -> Output
    
    The attention mechanism allows the model to focus on relevant
    past time steps when making predictions.
    """
    def __init__(self, input_size, hidden_size, num_layers=2, 
                 attention_type='bahdanau', attention_dim=64,
                 dropout=0.2, forecast_horizon=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.attention_type = attention_type
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size, attention_dim)
        else:
            self.attention = LuongAttention(hidden_size, method='general')
        
        # Output layers
        self.fc_context = nn.Linear(hidden_size * 2, hidden_size)  # Combine context + hidden
        self.fc_out = nn.Linear(hidden_size, input_size * forecast_horizon)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
    
    def forward(self, x):
        """
        Forward pass with attention.
        
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            output: (batch, input_size) or (batch, forecast_horizon, input_size)
        """
        batch_size = x.size(0)
        
        # Encode input sequence
        encoder_outputs, (hidden, cell) = self.encoder(x)
        # encoder_outputs: (batch, seq_len, hidden_size)
        # hidden: (num_layers, batch, hidden_size)
        
        # Use last layer hidden state for attention
        decoder_hidden = hidden[-1]  # (batch, hidden_size)
        
        # Apply attention
        context, attention_weights = self.attention(encoder_outputs, decoder_hidden)
        
        # Store for visualization
        self.attention_weights = attention_weights.detach()
        
        # Combine context and hidden state
        combined = torch.cat([context, decoder_hidden], dim=-1)
        combined = self.dropout(torch.relu(self.fc_context(combined)))
        
        # Generate output
        output = self.fc_out(combined)
        
        if self.forecast_horizon > 1:
            output = output.view(batch_size, self.forecast_horizon, -1)
        
        return output
    
    def get_attention_weights(self):
        """Return the attention weights from the last forward pass."""
        return self.attention_weights


class BaselineLSTM(nn.Module):
    """
    Standard LSTM without Attention (for baseline comparison).
    """
    def __init__(self, input_size, hidden_size, num_layers=2, 
                 dropout=0.2, forecast_horizon=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, input_size * forecast_horizon)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        out, (hidden, cell) = self.lstm(x)
        last_hidden = hidden[-1]
        output = self.fc(last_hidden)
        
        if self.forecast_horizon > 1:
            output = output.view(batch_size, self.forecast_horizon, -1)
        
        return output


# Backward compatibility
RNNModel = AttentionLSTM
LSTMModel = AttentionLSTM


if __name__ == "__main__":
    # Test both models
    batch, seq, features = 32, 10, 5
    x = torch.randn(batch, seq, features)
    
    print("Testing Attention LSTM (Bahdanau):")
    model = AttentionLSTM(input_size=features, hidden_size=64, num_layers=2, 
                          attention_type='bahdanau', attention_dim=32)
    out = model(x)
    print(f"  Output shape: {out.shape}")
    print(f"  Attention weights shape: {model.get_attention_weights().shape}")
    
    print("\nTesting Attention LSTM (Luong):")
    model2 = AttentionLSTM(input_size=features, hidden_size=64, num_layers=2,
                           attention_type='luong')
    out2 = model2(x)
    print(f"  Output shape: {out2.shape}")
    
    print("\nTesting Baseline LSTM (No Attention):")
    baseline = BaselineLSTM(input_size=features, hidden_size=64, num_layers=2)
    out3 = baseline(x)
    print(f"  Output shape: {out3.shape}")
