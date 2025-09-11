import requests
import json

# Test the exact same composition
test_data = {
    'compositions': {
        '1': 0.04,
        '2': 0.59, 
        '3': 0.87,
        '4': 0.41,
        '5': 0.441
    }
}

print("Testing Flask prediction endpoint...")
print(f"Sending data: {test_data}")

try:
    response = requests.post('http://127.0.0.1:5000/predict', 
                           headers={'Content-Type': 'application/json'},
                           json=test_data,
                           timeout=10)
    
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Full response: {json.dumps(result, indent=2)}")
        
        # Focus on Property5
        prop5_result = result.get('BlendProperty5', {})
        print(f"\nProperty5 specific result:")
        print(f"  Prediction: {prop5_result.get('prediction', 'Not found')}")
        print(f"  Model: {prop5_result.get('model', 'Not found')}")
        print(f"  R2 Score: {prop5_result.get('r2_score', 'Not found')}")
        print(f"  Error: {prop5_result.get('error', 'None')}")
        
    else:
        print(f"Error response: {response.text}")
        
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to Flask server. Is it running on port 5000?")
except requests.exceptions.Timeout:
    print("ERROR: Request timed out")
except Exception as e:
    print(f"ERROR: {e}")