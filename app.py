from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__)
CORS(app)

# Load all models, scalers, and performance data at startup
models = {}
scalers = {}
performance_metrics = {}
component_properties_df = None

try:
    # Load the new performance metrics file
    with open('model_performance.json', 'r') as f:
        performance_metrics = json.load(f)
        
    component_properties_df = pd.read_csv('component_properties.csv').set_index('Component')
    for i in range(1, 11):
        prop = f'BlendProperty{i}'
        with open(f'{prop}_model.pkl', 'rb') as f:
            models[prop] = pickle.load(f)
        scaler_path = f'{prop}_scaler.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scalers[prop] = pickle.load(f)
    print("✅ All models, scalers, and performance data loaded successfully.")
except Exception as e:
    print(f"❌ Error during server startup: {e}")
    print("-> Please ensure you have run the updated train_models.py script first.")

@app.route('/get-component-properties', methods=['POST'])
def get_component_properties():
    # This function remains unchanged
    data = request.get_json()
    component_id = int(data['component'])
    try:
        properties = component_properties_df.loc[component_id].to_dict()
        return jsonify(properties)
    except KeyError:
        return jsonify({'error': 'Component not found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    compositions = data['compositions']
    
    feature_vector = []
    # Part 1: Fractions
    for i in range(1, 6):
        feature_vector.append(float(compositions.get(str(i), 0)))
    # Part 2: Weighted properties
    for i in range(1, 11):
        for j in range(1, 6):
            fraction = float(compositions.get(str(j), 0))
            raw_prop = component_properties_df.loc[j, f'Property{i}']
            feature_vector.append(fraction * raw_prop)
    
    X_input = np.array(feature_vector).reshape(1, -1)
    
    predictions = {}
    for prop, model in models.items():
        X_to_predict = X_input
        if prop in scalers:
            X_to_predict = scalers[prop].transform(X_to_predict)
        prediction = model.predict(X_to_predict)
        
        # Get performance info for this specific property
        perf_info = performance_metrics.get(prop, {})

        # --- THIS IS THE UPGRADED RESPONSE ---
        # Include prediction, model name, and R² score
        predictions[prop] = {
            "prediction": max(0, prediction[0]),
            "model": perf_info.get("model", "N/A"),
            "r2_score": perf_info.get("r2_score", 0)
        }
        
    return jsonify(predictions)

if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    app.run(host='0.0.0.0', port=5000)

