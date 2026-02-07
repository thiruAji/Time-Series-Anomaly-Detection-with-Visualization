# nas_search.py
import itertools
import numpy as np

def nas_search(train_fn, X_train, y_train, X_val, y_val):
    """Neural Architecture Search over hyperparameter space
    
    Args:
        train_fn: Training function that takes (layers, hidden, X_train, y_train, X_val, y_val)
        X_train, y_train, X_val, y_val: Training and validation data
    
    Returns:
        Tuple of (best_config, best_score)
    """
    search_space = {
        "layers": [1, 2, 3],
        "hidden": [32, 64, 128]
    }

    best_config = None
    best_score = float("inf")

    for layers, hidden in itertools.product(
            search_space["layers"], search_space["hidden"]):
        
        print(f"Training with layers={layers}, hidden={hidden}")
        model, rmse = train_fn(layers, hidden, X_train, y_train, X_val, y_val)

        if rmse < best_score:
            best_score = rmse
            best_config = (layers, hidden)
            print(f"  New best RMSE: {rmse:.4f}")

    return best_config, best_score
