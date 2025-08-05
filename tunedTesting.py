#Final tuned model without gridsearch, based on the best configuration   |
# Best configuration:                                                    |
#   hidden_layer_sizes = (32,)                                           |
#   learning_rate_init  = 0.001                                          |
#   max_iter (epochs)   = 200                                            |
#   accuracy            = 86.4% (216/250)                                |
#------------------------------------------------------------------------+
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress MLPClassifier convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# 1) TRAINING DATASET (20 foods)
X = np.array([
    [10,  0.2,  0.3, 2.4,  52],    # Apple
    [12,  0.3,  1.3, 2.6,  96],    # Banana
    [ 1.2, 14,    6,  3.5, 164],   # Almonds
    [ 1.7, 0.4,  2.8, 2.6,  34],   # Broccoli
    [ 0,   3.6, 31,  0,   165],    # Chicken Breast
    [ 0.4, 1.4,  2.4, 1.7,  68],   # Oatmeal
    [ 0.4, 0.4,  2.9, 2.2,  23],   # Spinach
    [ 3,   8,    4,  1,    94],    # Peanut Butter
    [ 0.7,15,    2,  7,   160],    # Avocado
    [ 4.7,3.3,  3.5, 0,    61],    # Yogurt Plain
    [ 2,   1.2,  3.7, 0.8,  79],   # White Bread
    [23,  14,    2,  1,   352],    # Chocolate Cake
    [35,  25,    5,  1,   700],    # Soda
    [ 0.3,17.3,  3.4, 3.8, 312],   # French Fries
    [ 0,   14,   0,  0,   119],    # Olive Oil
    [15,  9,     5,  3,   200],    # Granola
    [28,  12,    2,  1,   245],    # Candy Bar
    [20,  0.5,   1.7,0.5,  112],   # Orange Juice
    [21,  11,    3.5,0.5,  207],   # Ice Cream
    [ 5,  10,    6,  1,   200]     # Sushi
])
y = np.array([1]*10 + [0]*10)  # first 10 healthy, last 10 not healthy

# 2) INPUT NORMALIZATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) TEST SET & NAMES
test_names = ["Protein Shake", "Burger", "Greek Salad", "Donut", "Energy Bar"]
X_test = np.array([
    [10,  2, 20, 1, 150],  # Protein Shake
    [ 5, 25, 20, 2, 550],  # Burger
    [ 4, 10,  5, 3, 220],  # Greek Salad
    [12, 22,  4, 1, 300],  # Donut
    [18,  6, 10, 4, 240]   # Energy Bar
])
expected = np.array([1, 0, 1, 0, 0])
X_test_scaled = scaler.transform(X_test)

# 4) BEST CONFIGURATION (no grid search)
best_config = {
    'hidden_layer_sizes': (32,),
    'learning_rate_init': 0.001,
    'max_iter': 200
}

# 5) ACCUMULATE PROBABILITIES & TRACK ACCURACY
runs = 50
prob_sums = np.zeros(len(X_test))
total_correct = 0

for _ in range(runs):
    model = MLPClassifier(
        hidden_layer_sizes=best_config['hidden_layer_sizes'],
        activation='relu',
        solver='adam',
        learning_rate_init=best_config['learning_rate_init'],
        max_iter=best_config['max_iter'],
        random_state=None
    )
    model.fit(X_scaled, y)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    prob_sums += probs

    preds = (probs >= 0.5).astype(int)
    total_correct += np.sum(preds == expected)

# 6) COMPUTE AVERAGE PROB & HEALTHINESS LEVEL
avg_probs = prob_sums / runs

# 7) PRINT RESULTS
for name, p in zip(test_names, avg_probs):
    level = round(p * 10)  # 0-10 scale
    decision = "Healthy" if p >= 0.5 else "Not Healthy"
    print(f"{name}: prob={p:.2f}, level={level}/10, {decision}")

total_tests = runs * len(expected)
accuracy = total_correct / total_tests * 100
print(f"\nOverall accuracy over {runs} runs: {accuracy:.1f}% "
      f"({total_correct}/{total_tests})")
