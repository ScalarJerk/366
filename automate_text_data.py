import pandas as pd
import os

# Directory paths
input_dir = "IRDS/RawDataCopy"
output_dir = "IRDS/CSVData"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all .txt files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace(".txt", ".csv"))

        # Extract metadata from the file name
        file_parts = filename.replace(".txt", "").split("_")
        SubjectID = file_parts[0]
        DateID = file_parts[1]
        GestureLabel = file_parts[2]
        RepetitionNumber = file_parts[3]
        CorrectLabel = file_parts[4]
        Position = file_parts[5]

        # Initialize an empty list to hold expanded rows
        expanded_rows = []

        # Read the file line by line
        with open(input_file, "r") as file:
            # Read the first line to check for version header
            first_line = file.readline().strip()
            
            # Check if the first line starts with "Version" or isn't in expected CSV format
            if first_line.startswith("Version") or "," not in first_line:
                # Skip the first line and process the rest
                lines = file.readlines()
            else:
                # Include the first line in processing
                lines = [first_line] + file.readlines()
            
            # Process all valid lines
            for line in lines:
                # Split the main fields from the joint data
                parts = line.strip().split(",", maxsplit=3)
                timestamp1 = int(parts[0])
                timestamp2 = int(parts[1])
                timestamp3 = int(parts[2])
                joint_data = parts[3]

                # Extract each joint tuple from the joint data
                joint_entries = joint_data.strip("()").split("),(")
                for joint_entry in joint_entries:
                    # Parse the joint tuple
                    joint = joint_entry.replace("(", "").replace(")", "").split(",")
                    joint_name = joint[0]
                    tracking_status = joint[1]
                    x = float(joint[2])
                    y = float(joint[3])
                    z = float(joint[4])
                    x_proj = float(joint[5])
                    y_proj = float(joint[6])
                    # Add the expanded row to the list
                    expanded_rows.append([SubjectID, DateID, GestureLabel, RepetitionNumber, CorrectLabel, Position, timestamp1, timestamp2, timestamp3, joint_name, tracking_status, x, y, z, x_proj, y_proj])

        # Create a DataFrame from the expanded rows
        df = pd.DataFrame(expanded_rows, columns=['SubjectID', 'DateID', 'GestureLabel', 'RepetitionNumber', 'CorrectLabel', 'Position', 'Timestamp1', 'Timestamp2', 'Timestamp3', 'JointName', 'TrackingStatus', 'X', 'Y', 'Z', 'X_proj', 'Y_proj'])

        # Save the DataFrame to a CSV file
        df.to_csv(output_file, index=False)
