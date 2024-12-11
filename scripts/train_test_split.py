#creating train_val_csv.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_val_csv(root_dir, output_dir, train_csv, val_csv, test_size=0.3, random_seed=42):
    
    data = []
    
    # Traverse through each subject folder
    for subject_folder in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, subject_folder)
        if os.path.isdir(subject_path):
            # Extract Correct Label from folder name
            parts = subject_folder.split("_")
            if len(parts) < 6:
                print(f"Skipping folder '{subject_folder}' - insufficient parts in name.")
                continue
            try:
                correct_label = int(parts[4])  # 5th element, index 4
            except ValueError:
                print(f"Skipping folder '{subject_folder}' - Correct Label is not an integer.")
                continue
            
            # Traverse through frame folders inside subject folder
            for frame_folder in os.listdir(subject_path):
                frame_path = os.path.join(subject_path, frame_folder)
                if os.path.isdir(frame_path):
                    # Find the .png file in the frame folder
                    for file in os.listdir(frame_path):
                        if file.endswith(".png"):
                            image_path = os.path.join(frame_path, file)
                            # Optionally, verify corresponding .csv exists
                            csv_file = file.replace(".png", ".csv")
                            csv_path = os.path.join(frame_path, csv_file)
                            if not os.path.exists(csv_path):
                                print(f"Warning: CSV file '{csv_path}' not found for image '{image_path}'. Skipping.")
                                continue
                            
                            # Append to data list
                            data.append({
                                "image_path": image_path,
                                "label": correct_label
                            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    print(f"Total samples collected: {len(df)}")
    
    # Shuffle and split
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=random_seed, shuffle=True, stratify=df['label'])
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to CSV
    train_csv_path = os.path.join(output_dir, train_csv)
    val_csv_path = os.path.join(output_dir, val_csv)
    
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"Train CSV saved at: {train_csv_path}")
    print(f"Validation CSV saved at: {val_csv_path}")

if __name__ == "__main__":
    # Define paths
    root_dir = "/Users/sakshiarjun/Desktop/366/DepthImages2-3"  # Replace with your actual path
    output_dir = "366/datasets"  
    train_csv = "train.csv"
    val_csv = "val.csv"
    
    # Create CSV files for training and validation
    create_train_val_csv(root_dir, output_dir, train_csv, val_csv, test_size=0.3, random_seed=42)
