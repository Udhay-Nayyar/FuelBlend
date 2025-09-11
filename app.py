from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import json

app = Flask(__name__, static_url_path='')
CORS(app)

@app.route('/')
@app.route('/index.html')
def serve_index():
    return send_file('index.html', mimetype='text/html')

@app.route('/composition')
@app.route('/composition.html')
def serve_composition():
    return send_file('composition.html', mimetype='text/html')

@app.route('/results')
@app.route('/results.html')
def serve_results():
    return send_file('results.html', mimetype='text/html')

@app.route('/insights')
@app.route('/insights.html')
def serve_insights():
    return send_file('insights.html', mimetype='text/html')

# Handle static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_file(filename)

# Get port from environment variable or default to 5000
port = int(os.environ.get('PORT', 5000))

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
    pass
except Exception as e:
    print(f"Error during server startup: {e}")
    print("Please ensure you have run the updated train_models.py script first.")

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "Server is working!"})

@app.route('/properties')
def properties():
    try:
        component_id = int(request.args.get('component', 1))
        composition_value = float(request.args.get('composition', 0.5))
        
        # Validate inputs
        if component_id < 1 or component_id > 5:
            return jsonify({"error": "Component ID must be between 1 and 5"}), 400
        if composition_value < 0 or composition_value > 1:
            return jsonify({"error": "Composition must be between 0 and 1"}), 400
        if component_properties_df is None:
            return jsonify({"error": "Component properties not loaded"}), 500
        
        # Get all properties for the component from the CSV data
        raw_properties = component_properties_df.loc[component_id].to_dict()
        result = {}
        for prop_name, prop_value in raw_properties.items():
            result[prop_name] = prop_value * composition_value
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
    except KeyError:
        return jsonify({"error": f"Component {component_id} not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/simple-properties')
def simple_properties():
    try:
        component_id = int(request.args.get('component', 1))
        composition_value = float(request.args.get('composition', 0.5))
        
        # Validate inputs
        if component_id < 1 or component_id > 5:
            return jsonify({"error": "Component ID must be between 1 and 5"}), 400
        if composition_value < 0 or composition_value > 1:
            return jsonify({"error": "Composition must be between 0 and 1"}), 400
        if component_properties_df is None:
            return jsonify({"error": "Component properties not loaded"}), 500
        
        raw_properties = component_properties_df.loc[component_id].to_dict()
        result = {}
        for prop_name, prop_value in raw_properties.items():
            result[prop_name] = prop_value * composition_value
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": f"Invalid input format: {str(e)}"}), 400
    except KeyError:
        return jsonify({"error": f"Component {component_id} not found"}), 404
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/get-component-properties', methods=['POST', 'GET'])
def get_component_properties():
    try:
        if request.method == 'GET':
            # Handle GET request with query parameters
            component_id = int(request.args.get('component', 1))
            composition_value = float(request.args.get('composition', 0.5))
        else:
            # Handle POST request with JSON
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Invalid JSON data'}), 400
            component_id = int(data.get('component', 1))
            composition_value = float(data.get('composition', 0))
        
        # Validate component ID
        if component_id not in [1, 2, 3, 4, 5]:
            return jsonify({'error': f'Invalid component ID: {component_id}. Must be 1-5.'}), 400
        
        # Validate composition value
        if composition_value < 0 or composition_value > 1:
            return jsonify({'error': f'Invalid composition value: {composition_value}. Must be between 0 and 1.'}), 400
        
        # Check if component properties are loaded
        if component_properties_df is None:
            return jsonify({'error': 'Component properties not loaded'}), 500
        
        # Get raw properties for the component
        raw_properties = component_properties_df.loc[component_id].to_dict()
        
        # Calculate composition-dependent properties
        result_properties = {}
        for prop_name, prop_value in raw_properties.items():
            # Show property value scaled by composition
            scaled_value = prop_value * composition_value
            result_properties[prop_name] = scaled_value
            
        return jsonify(result_properties)
        
    except ValueError as e:
        return jsonify({'error': f'Invalid input format: {str(e)}'}), 400
    except KeyError:
        return jsonify({'error': f'Component {component_id} not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'compositions' not in data:
            return jsonify({'error': 'Invalid request: missing compositions data'}), 400
        
        compositions = data['compositions']
        
        # Validate compositions
        if not compositions or not isinstance(compositions, dict):
            return jsonify({'error': 'Invalid compositions format'}), 400
        
        # Check if at least one component is selected
        total_composition = sum(float(compositions.get(str(i), 0)) for i in range(1, 6))
        if total_composition == 0:
            return jsonify({'error': 'No components selected or all compositions are zero'}), 400
        
        # Normalize compositions to ensure they sum to 1 (handle user input variations)
        # Keep zeros as zeros, only normalize non-zero values
        normalized_compositions = {}
        non_zero_total = 0
        
        # First pass: identify non-zero components and their total
        for i in range(1, 6):
            original_value = float(compositions.get(str(i), 0))
            if original_value > 0:
                non_zero_total += original_value
        
        # Second pass: normalize only non-zero components
        for i in range(1, 6):
            original_value = float(compositions.get(str(i), 0))
            if original_value > 0 and non_zero_total > 0:
                normalized_compositions[str(i)] = original_value / non_zero_total
            else:
                normalized_compositions[str(i)] = 0
        
        feature_vector = []
        
        # Part 1: Fractions (always include all 5 components, even if 0)
        for i in range(1, 6):
            fraction = normalized_compositions[str(i)]
            feature_vector.append(fraction)
        
        # Part 2: Weighted properties - MUST match training data order!
        # Training expects: Component1_Property1, Component2_Property1, ..., Component5_Property1,
        #                   Component1_Property2, Component2_Property2, ..., Component5_Property2, etc.
        for i in range(1, 11):  # For each property (1-10)
            for j in range(1, 6):  # For each component (1-5)
                fraction = normalized_compositions[str(j)]
                try:
                    raw_prop = component_properties_df.loc[j, f'Property{i}']
                    weighted_prop = fraction * raw_prop
                    feature_vector.append(weighted_prop)
                except KeyError:
                    return jsonify({'error': f'Component {j} or Property{i} not found in data'}), 500
        
        if len(feature_vector) != 55:
            return jsonify({'error': f'Invalid feature vector length: {len(feature_vector)}, expected 55'}), 500
        
        X_input = np.array(feature_vector).reshape(1, -1)
        

    
        predictions = {}
        
        # Check if models are loaded
        if not models:
            return jsonify({'error': 'Models not loaded. Please restart the server.'}), 500
        
        for prop, model in models.items():
            try:
                X_to_predict = X_input.copy()
                
                # Apply scaling if scaler exists for this property
                if prop in scalers:
                    X_to_predict = scalers[prop].transform(X_to_predict)
                
                # Make prediction
                prediction = model.predict(X_to_predict)
                

                
                # Validate prediction
                if prediction is None or len(prediction) == 0:
                    pred_value = 0.0
                else:
                    pred_value = float(prediction[0])
                
                # Get performance info for this specific property
                perf_info = performance_metrics.get(prop, {})

                predictions[prop] = {
                    "prediction": pred_value,
                    "model": perf_info.get("model", "N/A"),
                    "r2_score": perf_info.get("r2_score", 0)
                }
                
            except Exception as e:
                predictions[prop] = {
                    "prediction": 0.0,
                    "model": "Error",
                    "r2_score": 0,
                    "error": str(e)
                }
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask Server...")
    print(f"Server available at: http://0.0.0.0:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)

