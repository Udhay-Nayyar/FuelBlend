import pandas as pd
import numpy as np
import pickle
import json
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print("--- Starting Model Training & Performance Evaluation ---")

# --- 1. Load Data and Define Features ---
try:
    df = pd.read_csv("MIX.csv")
    print("✅ Dataset 'MIX.csv' loaded.")
except FileNotFoundError:
    print("❌ Error: 'MIX.csv' not found.")
    exit()

feature_columns = [
    'Component1_fraction', 'Component2_fraction', 'Component3_fraction', 'Component4_fraction', 'Component5_fraction',
    'Component1_Property1', 'Component2_Property1', 'Component3_Property1', 'Component4_Property1', 'Component5_Property1',
    'Component1_Property2', 'Component2_Property2', 'Component3_Property2', 'Component4_Property2', 'Component5_Property2',
    'Component1_Property3', 'Component2_Property3', 'Component3_Property3', 'Component4_Property3', 'Component5_Property3',
    'Component1_Property4', 'Component2_Property4', 'Component3_Property4', 'Component4_Property4', 'Component5_Property4',
    'Component1_Property5', 'Component2_Property5', 'Component3_Property5', 'Component4_Property5', 'Component5_Property5',
    'Component1_Property6', 'Component2_Property6', 'Component3_Property6', 'Component4_Property6', 'Component5_Property6',
    'Component1_Property7', 'Component2_Property7', 'Component3_Property7', 'Component4_Property7', 'Component5_Property7',
    'Component1_Property8', 'Component2_Property8', 'Component3_Property8', 'Component4_Property8', 'Component5_Property8',
    'Component1_Property9', 'Component2_Property9', 'Component3_Property9', 'Component4_Property9', 'Component5_Property9',
    'Component1_Property10', 'Component2_Property10', 'Component3_Property10', 'Component4_Property10', 'Component5_Property10'
]
X = df[feature_columns]

# --- 2. Updated Training Function ---
def train_and_evaluate_regressor(X, y, model, scale_input=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = None
    if scale_input:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    model.fit(X_train, y_train)
    # R-squared is a common metric for regression, returned by .score()
    r2_score = model.score(X_test, y_test)
    print(f"-> Model for '{y.name}' trained. Test Score (R²): {r2_score:.4f}")
    return model, scaler, r2_score

# --- 3. Model Configurations ---
model_configs = [
    {'target': 'BlendProperty1', 'model': SVR(), 'scale': True},
    {'target': 'BlendProperty2', 'model': SVR(), 'scale': True},
    {'target': 'BlendProperty3', 'model': RandomForestRegressor(random_state=0), 'scale': False},
    {'target': 'BlendProperty4', 'model': linear_model.LinearRegression(), 'scale': False},
    {'target': 'BlendProperty5', 'model': RandomForestRegressor(random_state=0), 'scale': False},
    {'target': 'BlendProperty6', 'model': SVR(), 'scale': True},
    {'target': 'BlendProperty7', 'model': RandomForestRegressor(random_state=0), 'scale': False},
    {'target': 'BlendProperty8', 'model': RandomForestRegressor(random_state=0), 'scale': False},
    {'target': 'BlendProperty9', 'model': RandomForestRegressor(random_state=0), 'scale': False},
    {'target': 'BlendProperty10', 'model': SVR(), 'scale': True}
]

# --- 4. Loop, Train, Save Models, and Collect Performance Data ---
performance_data = {}
print("\n--- Training and saving models... ---")
for config in model_configs:
    target_name = config['target']
    model_instance = config['model']
    
    y = df[target_name]
    trained_model, trained_scaler, r2 = train_and_evaluate_regressor(
        X, y, model_instance, scale_input=config['scale']
    )

    # Save the performance data
    performance_data[target_name] = {
        "model": type(trained_model).__name__,
        "r2_score": r2
    }
    
    # Save the model and scaler
    with open(f'{target_name}_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    if trained_scaler:
        with open(f'{target_name}_scaler.pkl', 'wb') as f:
            pickle.dump(trained_scaler, f)

# --- 5. Save Performance Data to a JSON file ---
with open('model_performance.json', 'w') as f:
    json.dump(performance_data, f, indent=4)
print("\n✅ Performance metrics saved to 'model_performance.json'")
print("✅ --- All models have been trained and saved successfully! ---")
