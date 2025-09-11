import pandas as pd
import numpy as np
import pickle

# Load the component properties and model
component_properties_df = pd.read_csv('component_properties.csv').set_index('Component')
with open('BlendProperty5_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Test composition from your screenshot
compositions = {
    '1': 0.04,
    '2': 0.59, 
    '3': 0.87,
    '4': 0.41,
    '5': 0.441
}

# Normalize compositions (same as Flask app)
total_composition = sum(float(compositions.get(str(i), 0)) for i in range(1, 6))
print(f"Total composition: {total_composition}")

normalized_compositions = {}
for i in range(1, 6):
    original_value = float(compositions.get(str(i), 0))
    normalized_compositions[str(i)] = original_value / total_composition if total_composition > 0 else 0

print(f"Normalized compositions: {normalized_compositions}")

# Build feature vector (same as Flask app)
feature_vector = []

# Part 1: Fractions
for i in range(1, 6):
    fraction = normalized_compositions[str(i)]
    feature_vector.append(fraction)

print(f"Fractions: {feature_vector}")

# Part 2: Weighted properties
for i in range(1, 11):  # For each property (1-10)
    for j in range(1, 6):  # For each component (1-5)
        fraction = normalized_compositions[str(j)]
        raw_prop = component_properties_df.loc[j, f'Property{i}']
        weighted_prop = fraction * raw_prop
        feature_vector.append(weighted_prop)

print(f"Feature vector length: {len(feature_vector)}")
print(f"Feature vector first 10: {feature_vector[:10]}")
print(f"Feature vector last 10: {feature_vector[-10:]}")

# Make prediction
X_input = np.array(feature_vector).reshape(1, -1)
prediction = model.predict(X_input)

print(f"Property5 prediction: {prediction[0]}")

# Let's also check what the training data looks like for comparison
df = pd.read_csv('MIX.csv')
print(f"\nTraining data Property5 stats:")
print(f"Min: {df['BlendProperty5'].min()}")
print(f"Max: {df['BlendProperty5'].max()}")
print(f"Mean: {df['BlendProperty5'].mean()}")

# Check a few training samples
print(f"\nFirst 5 training samples for Property5: {df['BlendProperty5'].head().tolist()}")