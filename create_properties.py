import pandas as pd

print("Reading the main dataset from MIX.csv...")

# Load your full dataset
try:
    df = pd.read_csv("MIX.csv")
except FileNotFoundError:
    print("Error: MIX.csv not found. Make sure it's in the same folder as this script.")
    exit()

# This dictionary will hold our extracted property data
properties_data = {
    'Component': [],
    'Property1': [], 'Property2': [], 'Property3': [], 'Property4': [], 'Property5': [],
    'Property6': [], 'Property7': [], 'Property8': [], 'Property9': [], 'Property10': []
}

# We assume the raw properties for a component are consistent across the dataset.
# So, we only need to read them once (e.g., from the first row).
print("Extracting raw properties for each of the 5 components...")
for component_num in range(1, 6):  # For components 1 through 5
    properties_data['Component'].append(component_num)
    for prop_num in range(1, 11):  # For properties 1 through 10
        # Construct the column name, e.g., 'Component1_Property1'
        column_name = f'Component{component_num}_Property{prop_num}'
        
        # Extract the value from the first row of your MIX.csv
        # .iloc[0] gets the value from the very first record
        value = df[column_name].iloc[0]
        
        # Add the extracted value to our dictionary
        properties_data[f'Property{prop_num}'].append(value)

# Convert the dictionary to a pandas DataFrame
properties_df = pd.DataFrame(properties_data)

# Save the new, neatly formatted DataFrame to a CSV file
try:
    properties_df.to_csv('component_properties.csv', index=False)
    print("\n✅ Successfully created 'component_properties.csv'.")
    print("You only need to run this script once.")
except Exception as e:
    print(f"\n❌ Error saving file: {e}")