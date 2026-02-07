# nas_search.py
"""
Neural Architecture Search using Evolutionary Strategy.
Searches over: layers, hidden units, cell type (LSTM/GRU), dropout, learning rate.
"""
import numpy as np
import random
import copy
from typing import Dict, List, Tuple, Callable

# Search space definition
SEARCH_SPACE = {
    "num_layers": [1, 2, 3],
    "hidden_size": [32, 64, 128, 256],
    "cell_type": ["LSTM", "GRU"],
    "dropout": [0.0, 0.1, 0.2, 0.3],
    "learning_rate": [0.001, 0.005, 0.01]
}


def random_config() -> Dict:
    """Generate a random configuration from search space."""
    return {
        key: random.choice(values)
        for key, values in SEARCH_SPACE.items()
    }


def mutate_config(config: Dict, mutation_rate: float = 0.3) -> Dict:
    """Mutate a configuration by randomly changing some parameters."""
    new_config = copy.deepcopy(config)
    for key in SEARCH_SPACE:
        if random.random() < mutation_rate:
            new_config[key] = random.choice(SEARCH_SPACE[key])
    return new_config


def crossover(parent1: Dict, parent2: Dict) -> Dict:
    """Create child configuration by crossing over two parents."""
    child = {}
    for key in SEARCH_SPACE:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child


def evolutionary_search(
    train_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    population_size: int = 10,
    generations: int = 5,
    elite_ratio: float = 0.2,
    mutation_rate: float = 0.3,
    verbose: bool = True
) -> Tuple[Dict, float, List[Dict]]:
    """
    Evolutionary Neural Architecture Search.
    
    Args:
        train_fn: Training function(config, X_train, y_train, X_val, y_val) -> (model, rmse)
        X_train, y_train, X_val, y_val: Training and validation data
        population_size: Number of individuals per generation
        generations: Number of evolutionary generations
        elite_ratio: Fraction of top performers to keep
        mutation_rate: Probability of mutating each gene
        verbose: Print progress
    
    Returns:
        Tuple of (best_config, best_score, search_history)
    """
    history = []
    
    # Initialize population with random configs
    population = [random_config() for _ in range(population_size)]
    
    best_config = None
    best_score = float('inf')
    best_model = None
    
    for gen in range(generations):
        if verbose:
            print(f"\n=== Generation {gen + 1}/{generations} ===")
        
        # Evaluate all individuals
        scores = []
        for i, config in enumerate(population):
            try:
                model, rmse = train_fn(config, X_train, y_train, X_val, y_val)
                scores.append((config, rmse, model))
                
                if verbose:
                    print(f"  [{i+1}/{len(population)}] {config['cell_type']}-L{config['num_layers']}-H{config['hidden_size']} -> RMSE: {rmse:.4f}")
                
                # Track best
                if rmse < best_score:
                    best_score = rmse
                    best_config = config
                    best_model = model
                    if verbose:
                        print(f"    *** New best! ***")
                        
            except Exception as e:
                if verbose:
                    print(f"  [{i+1}] Config failed: {e}")
                scores.append((config, float('inf'), None))
        
        # Record history
        history.append({
            'generation': gen + 1,
            'best_score': best_score,
            'best_config': copy.deepcopy(best_config),
            'population_scores': [s[1] for s in scores]
        })
        
        # Selection: sort by score, keep elites
        scores.sort(key=lambda x: x[1])
        n_elites = max(2, int(population_size * elite_ratio))
        elites = [s[0] for s in scores[:n_elites]]
        
        # Create next generation
        new_population = elites.copy()
        
        while len(new_population) < population_size:
            if random.random() < 0.5 and len(elites) >= 2:
                # Crossover
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
            else:
                # Mutation of random elite
                parent = random.choice(elites)
                child = mutate_config(parent, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
    
    if verbose:
        print(f"\n=== Search Complete ===")
        print(f"Best Config: {best_config}")
        print(f"Best RMSE: {best_score:.4f}")
    
    return best_config, best_score, history, best_model


# Backward compatibility wrapper
def nas_search(train_fn, X_train, y_train, X_val, y_val):
    """Legacy wrapper for evolutionary search."""
    best_config, best_score, history, model = evolutionary_search(
        train_fn, X_train, y_train, X_val, y_val,
        population_size=6,
        generations=3,
        verbose=True
    )
    # Return in legacy format (tuple of key values)
    return (best_config['num_layers'], best_config['hidden_size']), best_score


if __name__ == "__main__":
    print("NAS Search Space:")
    for key, values in SEARCH_SPACE.items():
        print(f"  {key}: {values}")
    print(f"\nTotal configurations: {np.prod([len(v) for v in SEARCH_SPACE.values()])}")
