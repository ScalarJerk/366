import os
import pandas as pd

# Directory containing the CSV files
input_directory = 'IRDS/CSVData'
output_directory = 'IRDS/CSVData_encoded'

# Iterate over all files in the directory
for filename in os.listdir(input_directory):
    if filename.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(input_directory, filename)
        
        # Load the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Perform one-hot encoding on the specified columns
        df_encoded = pd.get_dummies(df, columns=['Position', 'JointName', 'TrackingStatus'])
        
        # Save the transformed DataFrame to a new CSV file
        output_path = os.path.join(output_directory, f'encoded_{filename}')
        df_encoded.to_csv(output_path, index=False)
        
        print(f'Encoded file saved to: {output_path}')