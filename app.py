#final working code
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

def train_model():
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
    y = np.array([1]*10 + [0]*10)  # first 10 healthy, last 10 not healthy

    # 2) INPUT NORMALIZATION
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3) BEST CONFIGURATION
    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        max_iter=200,
        random_state=42
    )
    model.fit(X_scaled, y)
    return scaler, model

def predict_food(scaler, model):
    try:
        sugar = float(input("Enter sugar (g): "))
        fat = float(input("Enter fat (g): "))
        protein = float(input("Enter protein (g): "))
        fiber = float(input("Enter fiber (g): "))
        calories = float(input("Enter calories: "))
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    features = np.array([[sugar, fat, protein, fiber, calories]])
    features_scaled = scaler.transform(features)
    prob = model.predict_proba(features_scaled)[0, 1]
    level = round(prob * 10)
    decision = "Healthy" if prob >= 0.5 else "Not Healthy"
    print(f"\nPredicted probability of healthy: {prob:.2f}")
    print(f"Healthiness level (0-10): {level}")
    print(f"Decision: {decision}\n")

def main():
    print("Training model...")
    scaler, model = train_model()
    print("Model trained. Enter nutritional info below.\n")
    while True:
        predict_food(scaler, model)
        cont = input("Try another? (y/n): ").strip().lower()
        if cont != 'y':
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
