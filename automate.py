import csv, re, os, shutil
data = ""
root_dir = 'DepthImages1-2-alpha'
for sub_dir_name in os.listdir(root_dir):
    with open(f'./{root_dir}/{sub_dir_name}/{sub_dir_name}.txt', 'r') as file:
        content = file.readlines()
        for i in range(1, len(content)):
            data = content[i]
            data_id = data.split(',')[0]
            # make a new directory in DepthImages1-2-alpha/106_18_0_1_1_stand/{data_id}/
            path = f'{root_dir}/{sub_dir_name}/{data_id}'
            # try:
            #     os.mkdir(path)
            #     print(f"Folder {path} created!")
            # except FileExistsError:
            #     print(f"Folder {path} already exists")
            try:
                shutil.move(f'{root_dir}/{sub_dir_name}/{data_id}.png',f'{root_dir}/{sub_dir_name}/{data_id}/{data_id}.png')
                print(f"Image {data_id} moved!")
            except FileExistsError:
                print(f"Image {data_id} already exists")
            except FileNotFoundError:
                print(f"Image {data_id} not found")
            # Extracting the first three numbers
            # first_three = data.split('(')[0].strip().split(',')
            # # Extracting the data within parentheses
            # matches = re.findall(r'\((.*?)\)', data)
            # Writing to CSV
            # try:
            #     with open(f'{root_dir}/{sub_dir_name}/{data_id}/{data_id}.csv', mode='w', newline='') as file:
            #         writer = csv.writer(file)
            #         # Write header
            #         writer.writerow(['ID1', 'ID2', 'ID3', 'Joint', 'TrackingState', 'X', 'Y', 'Z', 'X2', 'Y2'])
            #         # Write the first three IDs
            #         for match in matches:
            #             row = first_three + match.split(',')
            #             writer.writerow(row)
            # except FileExistsError:
            #     print(f"File {data_id}.csv already exists")
            # print(data_id)