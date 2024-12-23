import cv2
import numpy as np
import os
from pathlib import Path
import glob

def analyze_frames(input_dir, output_dir):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get list of image files
    image_files = sorted([f for f in os.listdir(input_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_files:
        print(f"No image files found in the input directory: {input_dir}")
        return None, None
    
    # Initialize variables
    prev_frame = cv2.imread(os.path.join(input_dir, image_files[0]))
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    key_frames = [prev_frame]  # Store key frames
    key_frame_names = [image_files[0]]  # Store key frame filenames
    
    # Process each frame
    for img_file in image_files[1:]:
        current_frame = cv2.imread(os.path.join(input_dir, img_file))
        
        if current_frame is None:
            print(f"Failed to read image: {img_file}")
            continue
            
        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        frame_diff = cv2.absdiff(prev_frame_gray, current_frame_gray)
        
        # Thresholding
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Count non-zero pixels
        non_zero_count = np.count_nonzero(thresh)
        
        # Define a threshold for significant motion
        if non_zero_count > 1000:  # Adjust this value based on your needs
            key_frames.append(current_frame)  # Save key frame
            key_frame_names.append(img_file)  # Save filename
            
        # Update previous frame
        prev_frame_gray = current_frame_gray
    
    # Save key frames
    for i, (frame, original_name) in enumerate(zip(key_frames, key_frame_names)):
        # Create output filename using original filename and index
        base_name = os.path.splitext(original_name)[0]
        output_path = os.path.join(output_dir, f'key_frame_{base_name}_{i}.jpg')
        cv2.imwrite(output_path, frame)
    
    print(f"Extracted {len(key_frames)} key frames from {input_dir}")
    return key_frames, key_frame_names

def process_all_directories(base_path="."):
    # Find all directories containing "copy" but not "final_copy"
    copy_dirs = []
    for root, dirs, files in os.walk(base_path):
        for dir_name in dirs:
            if "copy" in dir_name and "final_copy" not in dir_name:
                full_path = os.path.join(root, dir_name)
                copy_dirs.append(full_path)
    
    results = {}
    for input_dir in copy_dirs:
        # Create corresponding output directory path
        parent_dir = os.path.dirname(input_dir)
        output_dir = os.path.join(parent_dir, "final_copy")
        
        print(f"\nProcessing directory: {input_dir}")
        print(f"Output directory: {output_dir}")
        
        # Process the directory
        key_frames, key_frame_names = analyze_frames(input_dir, output_dir)
        
        if key_frames is not None:
            results[input_dir] = len(key_frames)
    
    # Print summary
    print("\nProcessing Summary:")
    print("-" * 50)
    for directory, num_frames in results.items():
        print(f"{directory}: {num_frames} key frames extracted")
    print(f"Total directories processed: {len(results)}")

if __name__ == "__main__":
    # You can specify a base path as an argument, or it will use the current directory
    base_path = "."  # or specify your base path here
    process_all_directories(base_path)
    
    
'''
Steps to Remove Redundant Frames
1. Load the Video or Image Sequence: Start by capturing the video or loading the sequence of images.
2. Convert Frames to Grayscale: Convert each frame to grayscale to simplify the comparison process.
3. Calculate Frame Differences: Compute the absolute difference between consecutive frames. This will highlight areas of motion.
4. Thresholding: Apply a threshold to the difference image to create a binary image where significant changes are marked.
5. Count Non-Zero Pixels: Count the number of non-zero pixels in the thresholded image. If this count exceeds a predefined threshold, consider it a key frame.
6. Store Key Frames: Save frames that meet the criteria for significant motion.
'''