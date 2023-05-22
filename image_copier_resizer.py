import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_resizer(image_path, output_file, resize):
    basename = os.path.basename(image_path)
    outpath = os.path.join(output_file, basename)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)

# Create the directory to save resized images
os.makedirs('resized_images', exist_ok=True)

# Deploy the function in a for loop
files = os.listdir('HAM10000_images_part1&2')
for file in files:
    file_path = os.path.join('HAM10000_images_part1&2', file)
    image_resizer(file_path, 'resized_images', [224, 224])



import shutil
import pandas as pd

df = pd.read_csv('train.csv')

def image_copier(source_folder, destination_folder):
    # Get the list of files in the source folder
    files = os.listdir(source_folder)

    # Iterate over each file in the source folder
    for file in files:
        # Get the basename of the file without the extension
        filename = os.path.splitext(file)[0]

        if filename in df['image_id'].values:
            # Create the source and destination paths
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)
            # Copy the file to the destination folder
            shutil.copyfile(source_path, destination_path)
            print(f"Image {file} copied successfully.")
