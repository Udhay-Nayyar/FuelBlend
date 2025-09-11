import pandas as pd
import numpy as np
import pickle
import json

# Load everything exactly like Flask does
component_properties_df = pd.read_csv('component_properties.csv').set_index('Component')

# Load models
models = {}
for i in range(1, 11):
    prop = f'BlendProperty{i}'
    with open(f'{prop}_model.pkl', 'rb') as f:
        models[prop] = pickle.load(f)

# Load scalers
scalers = {}
for i in range(1, 11):
    prop = f'BlendProperty{i}'
    scaler_path = f'{prop}_scaler.pkl'
    try:
        with open(scaler_path, 'rb') as f:
            scalers[prop] = pickle.load(f)
        print(f"Loaded scaler for {prop}")
    except FileNotFoundError:
        print(f"No scaler for {prop}")

# Load performance metrics
with open('model_performance.json', 'r') as f:
    performance_metrics = json.load(f)

print(f"Loaded {len(models)} models")
print(f"Loaded {len(scalers)} scalers")

# Test composition (exact same as Flask receives)
compositions = {
    '1': 0.04,
    '2': 0.59, 
    '3': 0.87,
    '4': 0.41,
    '5': 0.441
}

print(f"\nOriginal compositions: {compositions}")

# Replicate Flask logic exactly
total_composition = sum(float(compositions.get(str(i), 0)) for i in range(1, 6))
print(f"Total composition: {total_composition}")

# Normalize compositions
normalized_compositions = {}
for i in range(1, 6):
    original_value = float(compositions.get(str(i), 0))
    normalized_compositions[str(i)] = original_value / total_composition if total_composition > 0 else 0

print(f"Normalized compositions: {normalized_compositions}")

# Build feature vector exactly like Flask
feature_vector = []

# Part 1: Fractions
for i in range(1, 6):
    fraction = normalized_compositions[str(i)]
    feature_vector.append(fraction)

# Part 2: Weighted properties
for i in range(1, 11):  # For each property (1-10)
    for j in range(1, 6):  # For each component (1-5)
        fraction = normalized_compositions[str(j)]
        raw_prop = component_properties_df.loc[j, f'Property{i}']
        weighted_prop = fraction * raw_prop
        feature_vector.append(weighted_prop)

print(f"Feature vector length: {len(feature_vector)}")

X_input = np.array(feature_vector).reshape(1, -1)

# Test each model exactly like Flask does
print(f"\n=== Testing predictions ===")
for prop, model in models.items():
    try:
        X_to_predict = X_input.copy()
        
        # Apply scaling if scaler exists
        if prop in scalers:
            print(f"Applying scaler to {prop}")
            X_to_predict = scalers[prop].transform(X_to_predict)
        
        # Make prediction
        prediction = model.predict(X_to_predict)
        
        print(f"{prop}: {prediction[0]:.6f}")
        
        # Check if this matches what Flask would return
        if prediction is None or len(prediction) == 0:
            pred_value = 0.0
        else:
            pred_value = float(prediction[0])
            
        if pred_value == 0.0:
            print(f"  *** {prop} would return 0 in Flask! ***")
            
    except Exception as e:
        print(f"{prop}: ERROR - {e}")