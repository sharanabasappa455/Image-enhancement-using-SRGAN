import kagglehub
import shutil
import os
import glob

# 1. Settings
TARGET_FOLDER = "data/train_images"

# Create the folder if it doesn't exist
if not os.path.exists(TARGET_FOLDER):
    os.makedirs(TARGET_FOLDER)
    print(f"Created folder: {TARGET_FOLDER}")

# 2. Download Dataset
print("⬇️ Downloading BSDS300 dataset... (This may take a moment)")
path = kagglehub.dataset_download("adheshgarg/bsds300")
print(f"✅ Downloaded to cache: {path}")

# 3. Move Images to Project Folder
print("📂 Moving images to your project folder...")

# BSDS300 usually has a structure like 'images/train' inside. 
# We will search recursively for ALL .jpg files in the download.
image_files = glob.glob(os.path.join(path, "**/*.jpg"), recursive=True)

count = 0
for file_path in image_files:
    # Only move files, ignore directories
    if os.path.isfile(file_path):
        file_name = os.path.basename(file_path)
        destination = os.path.join(TARGET_FOLDER, file_name)
        
        # Copy file
        shutil.copy(file_path, destination)
        count += 1

print(f"🎉 Success! Moved {count} images into '{TARGET_FOLDER}'.")
print("You can now run 'python train.py'.")