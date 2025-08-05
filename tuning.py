#grid search so I can find the best configuration
import warnings
from sklearn.exceptions import ConvergenceWarning

# ──────────────────────────────────────────────────────────────
# Suppress MLPClassifier convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# ──────────────────────────────────────────────────────────────

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from itertools import product

# 1) TRAINING DATASET (20 foods)
X = np.array([
    [10,  0.2,  0.3, 2.4,  52],    # Apple
    [12,  0.3,  1.3, 2.6,  96],    # Banana
    [ 1.2,14,    6,  3.5, 164],    # Almonds
    [ 1.7,0.4,  2.8, 2.6,  34],    # Broccoli
    [ 0,   3.6, 31,  0,   165],    # Chicken Breast
    [ 0.4,1.4,  2.4, 1.7,  68],    # Oatmeal
    [ 0.4,0.4,  2.9, 2.2,  23],    # Spinach
    [ 3,   8,    4,  1,    94],    # Peanut Butter
    [ 0.7,15,    2,  7,   160],    # Avocado
    [ 4.7,3.3,  3.5, 0,    61],    # Yogurt Plain
    [ 2,   1.2,  3.7, 0.8,  79],    # White Bread
    [23,  14,    2,  1,   352],    # Chocolate Cake
    [35,  25,    5,  1,   700],    # Soda
    [ 0.3,17.3,  3.4, 3.8, 312],    # French Fries
    [ 0,   14,   0,  0,   119],    # Olive Oil
    [15,  9,     5,  3,   200],    # Granola
    [28,  12,    2,  1,   245],    # Candy Bar
    [20,  0.5,   1.7,0.5,  112],    # Orange Juice
    [21,  11,    3.5,0.5,  207],    # Ice Cream
    [ 5,  10,    6,  1,   200]     # Sushi
])
y = np.array([1]*10 + [0]*10)  # first 10 healthy, last 10 not

# 2) INPUT NORMALIZATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) TEST SET & EXPECTED
X_test = np.array([
    [10,  2, 20, 1, 150],  # Protein Shake
    [ 5, 25, 20, 2, 550],  # Burger
    [ 4, 10,  5, 3, 220],  # Greek Salad
    [12, 22,  4, 1, 300],  # Donut
    [18,  6, 10, 4, 240]   # Energy Bar
])
expected = np.array([1, 0, 1, 0, 0])
X_test_scaled = scaler.transform(X_test)

# 4) DEFINE GRID
param_grid = {
    'hidden_layer_sizes': [(4,),  (8,),  (16,),  (32,)],
    'learning_rate_init': [1e-2,  1e-3,  1e-4],
    'max_iter': [200,   500,   1000]
}

# 5) GRID SEARCH WITH 50 REPEATS, COUNTING INDIVIDUAL CORRECT PREDICTIONS
results = []
best = {'nodes': None, 'lr': None, 'epochs': None, 'correct': -1}

for hls, lr, mi in product(param_grid['hidden_layer_sizes'],
                           param_grid['learning_rate_init'],
                           param_grid['max_iter']):
    total_correct = 0
    for _ in range(50):
        model = MLPClassifier(
            hidden_layer_sizes=hls,
            activation='relu',
            solver='adam',
            learning_rate_init=lr,
            max_iter=mi,
            random_state=None
        )
        model.fit(X_scaled, y)
        preds = (model.predict_proba(X_test_scaled)[:,1] >= 0.5).astype(int)
        # Count how many of the 5 test cases this run got correct
        total_correct += np.sum(preds == expected)

    # Record this combo’s total correct out of 250
    results.append((hls[0], lr, mi, total_correct))

    # Update best if this combo has more total_correct
    if total_correct > best['correct']:
        best.update({
            'nodes': hls[0],
            'lr': lr,
            'epochs': mi,
            'correct': total_correct
        })

# 6) SHOW ALL RESULTS AS PERCENTAGES
print("nodes | lr    | epochs | accuracy (%)")
for nodes, lr, mi, correct in results:
    pct = correct / (50 * len(expected)) * 100
    print(f"{nodes:>5} | {lr:<5} | {mi:<6} | {pct:6.1f}%")

# 7) PRINT BEST CONFIGURATION
best_pct = best['correct'] / (50 * len(expected)) * 100
print("\nBest configuration:")
print(f"  hidden_layer_sizes = ({best['nodes']},)")
print(f"  learning_rate_init  = {best['lr']}")
print(f"  max_iter (epochs)   = {best['epochs']}")
print(f"  accuracy            = {best_pct:.1f}% ({best['correct']}/"
      f"{50 * len(expected)})")
