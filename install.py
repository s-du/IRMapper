import os
import requests
import zipfile
import tempfile
from tqdm import tqdm
import shutil

# Define the URL of the zip file
zip_url = "https://dl.djicdn.com/downloads/dji_thermal_sdk/20221108/dji_thermal_sdk_v1.4_20220929.zip"

# Define the target directory for extraction
target_dir = "resources/dji"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Create a temporary directory for extraction
temp_dir = tempfile.mkdtemp()

# Download the zip file
response = requests.get(zip_url, stream=True)
total_size = int(response.headers.get("content-length", 0))
block_size = 1024  # 1KB
progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

zip_path = os.path.join(temp_dir, "dji_thermal_sdk.zip")

with open(zip_path, "wb") as zip_file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        zip_file.write(data)

progress_bar.close()

# Extract the contents of the zip file to the temporary directory
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    for file in zip_ref.namelist():
        if file.startswith("utility/bin/linux/release_x64/") or file.startswith("utility/bin/windows/release_x64/"):
            source_path = zip_ref.extract(file, path=temp_dir)
            target_path = os.path.join(target_dir, os.path.basename(file))
            if os.path.isdir(source_path):  # Check if the extracted item is a directory
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                shutil.copy(source_path, target_path)

# Remove the zip file
os.remove(zip_path)

# Remove the temporary directory
shutil.rmtree(temp_dir)

print("Installation completed.")
