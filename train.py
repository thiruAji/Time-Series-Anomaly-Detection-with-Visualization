# train.py
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model import LSTMModel

def train_and_eval(layers, hidden, X_train, y_train, X_val, y_val, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine input size dynamically from data: (samples, time_steps, features)
    input_size = X_train.shape[2]
    model = LSTMModel(input_size=input_size, hidden_size=hidden, num_layers=layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val).to(device)
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)
        rmse = np.sqrt(val_loss.cpu().item())
    
    return model, rmse
