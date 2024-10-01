import csv, re, os, shutil

# Define the root directory containing all the subdirectories
root_dir = 'DepthImages1-2'

# Loop through all subdirectories in the root directory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    
    if os.path.isdir(subdir_path):
        # Find the text file in the subdirectory
        txt_file = os.path.join(root_dir, f"{subdir}.txt")
        
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as file:
                # Read each line in the text file
                for line in file:
                    # Extract image number (first element) and the rest of the data
                    first_part = line.split('(')[0].strip()  # Before the first '('
                    image_id = first_part.split(',')[0]  # Image ID is the first number
                    first_three = first_part.split(',')[:3]  # First three numbers
                    
                    # Find all the data in parentheses using regex
                    matches = re.findall(r'\((.*?)\)', line)

                    # Create a directory for the image (if not exists) named {image_id}
                    image_dir = os.path.join(subdir_path, image_id)
                    os.makedirs(image_dir, exist_ok=True)
                    
                    # Move the corresponding image file to this directory
                    image_file = os.path.join(subdir_path, f"{image_id}.png")
                    if os.path.exists(image_file):
                        shutil.move(image_file, image_dir)
                    
                    # Create a CSV file for this image
                    csv_file = os.path.join(image_dir, f"{image_id}.csv")
                    with open(csv_file, mode='w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        
                        # Write CSV headers
                        writer.writerow(['ID1', 'ID2', 'ID3', 'Joint', 'TrackingState', 'X', 'Y', 'Z', 'X2', 'Y2'])

                        # Write the data rows
                        for match in matches:
                            row = first_three + match.split(',')
                            writer.writerow(row)

                    print(f"Processed {image_id}.png and created {image_id}.csv in {image_dir}")

data = ""
root_dir = 'DepthImages1-2'
for sub_dir_name in os.listdir(root_dir):
    with open(f'./{root_dir}/{sub_dir_name}/{sub_dir_name}.txt', 'r') as file:
        content = file.readlines()
        for i in range(1, len(content)):
            data = content[i]
            data_id = data.split(',')[0]
            # make a new directory in DepthImages1-2-alpha/106_18_0_1_1_stand/{data_id}/
            path = f'{root_dir}/{sub_dir_name}/{data_id}'
            try:
                os.mkdir(path)
                print(f"Folder {path} created!")
            except FileExistsError:
                print(f"Folder {path} already exists")
            try:
                shutil.move(f'{root_dir}/{sub_dir_name}/{data_id}.png',f'{root_dir}/{sub_dir_name}/{data_id}/{data_id}.png')
                print(f"Image {data_id} moved!")
            except FileExistsError:
                print(f"Image {data_id} already exists")
            except FileNotFoundError:
                print(f"Image {data_id} not found")
            # Extracting the first three numbers
            first_three = data.split('(')[0].strip().split(',')
            # Extracting the data within parentheses
            matches = re.findall(r'\((.*?)\)', data)
            # Writing to CSV
            try:
                with open(f'{root_dir}/{sub_dir_name}/{data_id}/{data_id}.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    # Write header
                    writer.writerow(['ID1', 'ID2', 'ID3', 'Joint', 'TrackingState', 'X', 'Y', 'Z', 'X2', 'Y2'])
                    # Write the first three IDs
                    for match in matches:
                        row = first_three + match.split(',')
                        writer.writerow(row)
            except FileExistsError:
                print(f"File {data_id}.csv already exists")
            print(data_id)
