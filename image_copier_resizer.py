import os
import glob
from tqdm import tqdm
from PIL import Image, ImageFile
from joblib import Parallel, delayed

ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_resizer (image_path, output_file, resize):
    basename = os.path.basename(image_path)
    outpath = os.path.join(output_file, basename)
    img = Image.open(image_path)
    img = img.resize((resize[1], resize[0]), resample=Image.BILINEAR)
    img.save(outpath)

os.mkdir('train_images')
os.mkdir('test_images')

import os
import shutil
import pandas as pd

df = pd.read_csv('train.csv')

def copy_images(source_folder, destination_folder):
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

# Example usage
source_folder = 'HAM10000_images_part1&2'
destination_folder = 'train_images'

copy_images(source_folder, destination_folder)
