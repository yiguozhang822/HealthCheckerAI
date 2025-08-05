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

# 100-food training dataset (50 healthy, 50 unhealthy)
X = np.array([
    # Healthy foods (50)
    [13.8,  0.2,  0.3,  2.4,  52],   # Apple
    [12.2,  0.3,  1.1,  2.6,  96],   # Banana
    [10.0,  0.3,  0.7,  2.4,  57],   # Blueberries
    [ 4.9,  0.3,  0.8,  2.0,  32],   # Strawberries
    [ 9.4,  0.1,  0.9,  2.2,  47],   # Orange
    [16.0,  0.2,  0.6,  0.9,  69],   # Grapes
    [ 1.7,  0.4,  2.8,  2.6,  34],   # Broccoli
    [ 0.4,  0.4,  2.9,  2.2,  23],   # Spinach
    [ 0.8,  0.9,  2.0,  2.0,  35],   # Kale
    [ 4.7,  0.2,  0.9,  2.8,  41],   # Carrot
    [ 1.7,  0.1,  0.7,  0.5,  16],   # Cucumber
    [ 2.6,  0.2,  0.9,  1.2,  18],   # Tomato
    [ 4.2,  0.3,  1.0,  1.7,  31],   # Bell Pepper
    [ 0.7, 15.0,  2.0,  7.0, 160],   # Avocado
    [ 4.4, 49.9, 21.2, 12.5, 579],   # Almonds
    [ 1.0,  1.4,  2.4,  1.7,  68],   # Oatmeal
    [ 0.9,  1.9,  4.4,  2.8, 120],   # Quinoa
    [ 4.8,  2.6,  8.9,  7.6, 164],   # Chickpeas
    [ 1.9,  0.4,  9.0,  7.9, 116],   # Lentils
    [ 0.3,  0.5,  8.9,  8.7, 339],   # Black Beans
    [ 0.0, 13.4, 20.4,  0.0, 208],   # Salmon
    [ 0.0,  0.8, 29.9,  0.0, 132],   # Tuna
    [ 0.0,  3.6, 31.0,  0.0, 165],   # Chicken Breast
    [ 0.0,  1.2, 29.0,  0.0, 135],   # Turkey Breast
    [ 0.7, 10.6, 12.6,  0.0, 155],   # Egg (whole)
    [ 4.7,  3.3,  3.5,  0.0,  61],   # Greek Yogurt (plain)
    [ 3.4,  4.3, 11.1,  0.0,  98],   # Cottage Cheese
    [ 0.3,  4.8,  8.1,  0.3,  76],   # Tofu
    [ 4.2,  0.1,  1.6,  3.0,  86],   # Sweet Potato
    [ 0.4,  0.9,  2.6,  1.8, 111],   # Brown Rice
    [ 3.0,  0.2, 10.7,  0.0,  69],   # Quark
    [ 2.2,  5.2, 11.9,  5.2, 122],   # Edamame
    [ 0.9, 19.4, 30.2,  6.0, 559],   # Pumpkin Seeds
    [ 0.0, 31.4, 16.5, 34.4, 486],   # Chia Seeds
    [ 1.5, 42.2, 18.3, 27.3, 534],   # Flax Seeds
    [ 2.6, 65.2, 15.2,  6.7, 654],   # Walnuts
    [ 7.7, 45.0, 20.6, 10.6, 562],   # Pistachios
    [ 2.6, 51.5, 20.8,  8.6, 584],   # Sunflower Seeds
    [ 0.5,  6.9, 16.9, 10.6, 389],   # Oats
    [ 2.0,  2.0, 11.1,  4.0,  80],   # Quorn (mycoprotein)
    [ 2.2,  0.3,  3.4,  3.8,  43],   # Brussels Sprouts
    [ 1.9,  0.2,  2.2,  2.1,  20],   # Asparagus
    [ 0.5,  0.3,  3.1,  1.0,  22],   # Mushrooms
    [ 4.9,  0.5,  1.4,  5.3,  43],   # Blackberries
    [ 4.4,  0.7,  1.2,  6.5,  52],   # Raspberries
    [ 6.2,  0.2,  0.6,  0.4,  30],   # Watermelon
    [ 5.9,  0.3,  0.5,  1.7,  43],   # Papaya
    [ 9.0,  0.5,  1.1,  3.0,  61],   # Kiwi
    [ 3.0,  8.0,  4.0,  1.0,  94],   # Peanut Butter
    [ 5.0,  0.1,  3.4,  0.0,  34],   # Skim Milk

    # Unhealthy foods (50)
    [23.0, 14.0,  2.0,  1.0, 352],   # Chocolate Cake
    [10.6,  0.0,  0.0,  0.0,  42],   # Coca-Cola (100ml)
    [ 2.3, 28.0,  9.1,  3.0, 468],   # Cheez-It
    [ 0.3, 35.0,  7.0,  3.0, 547],   # Potato Chips
    [21.0, 11.0,  3.5,  0.5, 207],   # Ice Cream
    [21.0, 25.0,  4.0,  0.8, 452],   # Glazed Donut
    [48.0, 12.0,  4.0,  1.0, 488],   # Snickers Bar
    [47.0, 19.0,  4.4,  2.4, 480],   # Oreo Cookie
    [52.0, 30.0,  7.0,  2.8, 535],   # Milk Chocolate
    [ 3.9, 10.0, 12.0,  2.0, 266],   # Pizza (cheese)
    [ 6.0, 12.0, 17.0,  1.2, 295],   # Burger (beef)
    [ 0.3, 17.3,  3.4,  3.8, 312],   # French Fries
    [ 0.0, 19.2, 24.0,  0.0, 320],   # Fried Chicken
    [ 3.7, 12.0,  9.1,  2.5, 166],   # Lasagna
    [19.0, 10.0,  6.0,  1.2, 227],   # Pancakes w/ Syrup
    [ 8.0,  8.0,  6.3,  2.0, 291],   # Waffles
    [27.0, 14.0,  4.0,  1.5, 379],   # Muffin
    [ 5.6,  1.5,  9.0,  2.4, 250],   # Bagel
    [ 2.0,  1.2,  3.7,  0.8,  79],   # White Bread
    [ 4.8, 21.0,  8.0,  2.1, 406],   # Croissant
    [19.0, 11.0,  4.0,  0.9, 168],   # Ice Cream Cone
    [43.0, 20.0,  4.2,  1.1, 488],   # Chocolate Chip Cookie
    [20.0, 22.0,  2.0,  0.5, 275],   # Donut Hole
    [29.0, 12.0,  3.0,  1.5, 466],   # Brownie
    [10.0, 16.0,  7.0,  1.0, 275],   # Corn Dog
    [ 0.5, 15.0,  9.0,  1.3, 300],   # Mozzarella Sticks
    [ 3.8, 12.0,  5.0,  1.0, 190],   # Mac and Cheese
    [ 1.0, 21.0, 14.0,  0.0, 310],   # Spam
    [ 0.4, 42.0, 37.0,  0.0, 541],   # Bacon
    [ 1.3, 31.0, 14.0,  0.0, 339],   # Pork Sausage
    [ 5.0, 22.0, 11.0,  1.5, 290],   # Hot Dog
    [ 1.8, 18.0,  7.0,  2.2, 327],   # Poutine
    [ 1.7, 29.0,  7.0,  3.6, 536],   # Nachos
    [ 5.0, 12.0, 17.0,  1.2, 295],   # Cheeseburger
    [50.0,  9.0,  4.0,  0.0, 308],   # Milkshake
    [11.0,  0.0,  0.0,  0.0,  45],   # Energy Drink
    [18.0,  9.0,  5.0,  3.0, 200],   # Granola Bar
    [56.0, 31.0,  6.0,  2.1, 546],   # Chocolate Spread
    [20.0, 12.0,  3.0,  1.0, 200],   # Peanut Butter Cookie
    [ 0.4, 26.0,  7.0,  7.0, 487],   # Tortilla Chips
    [ 1.5, 19.0,  1.7,  3.0, 150],   # Guacamole (store-bought)
    [ 3.0,  5.0,  2.0,  0.5,  59],   # Canned Soup (cream)
    [ 1.4, 14.0,  5.0,  0.7, 436],   # Instant Ramen
    [18.0,  1.8,  2.0,  0.5, 106],   # Chocolate Pudding
    [24.0, 11.0,  3.0,  0.7, 205],   # Ice Cream Sandwich
    [28.0, 12.0,  2.0,  1.0, 245],   # Candy Bar
    [11.0,  0.0,  0.0,  0.0,  42],   # Pepsi (100ml)
    [11.0,  0.0,  0.0,  0.0,  45],   # Red Bull (100ml)
    [ 1.2,  3.3,  8.2,  3.4, 379],   # Pretzels
    [28.0, 12.0,  7.0,  1.0, 230]    # French Toast
])

y = np.array([1]*50 + [0]*50)

# 2) INPUT NORMALIZATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) TEST SET & EXPECTED
X_test = np.array([
    [ 2.0,  1.0, 20.0,  0.0, 100],  # Protein Shake Powder
    [ 3.9, 10.0, 12.0,  2.0, 266],  # Cheese Pizza Slice
    [18.0,  0.5,  2.0,  1.0,  90],  # Strawberry Smoothie
    [15.0, 15.0,  6.0,  2.0, 350],  # Chocolate Croissant
    [ 7.0,  5.0, 10.0,  0.0, 120]   # Greek Yogurt Bowl
])
expected = np.array([1, 0, 1, 0, 1])
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
