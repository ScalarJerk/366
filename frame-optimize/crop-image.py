from PIL import Image 
import os  
# open the image 
# ... existing code ...
# Create 'copy' directory if it doesn't exist

for i in os.listdir():
    # Only process if i is a directory
    if os.path.isdir(i):
        for j in os.listdir(i):  # Use os.listdir(i) instead of iterating string
            if j.lower().endswith('.png'):
                Image1 = Image.open(os.path.join(i, j))  # Use full path
                # crop the image 
                croppedIm = Image1.crop((100,10,400,400)) 
                
                # Create 'copy' directory if it doesn't exist
                copy_dir = os.path.join(i, 'copy')
                os.makedirs(copy_dir, exist_ok=True)
                
                # save the image in 'copy' folder with same name
                croppedIm.save(os.path.join(copy_dir, j))
