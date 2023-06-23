import os
import requests
import zipfile
from tqdm import tqdm

# Define the URL of the zip file
zip_url = "https://dl.djicdn.com/downloads/dji_thermal_sdk/20221108/dji_thermal_sdk_v1.4_20220929.zip"

# Define the target directory for extraction
target_dir = "resources/dji"

# Create the target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Download the zip file
response = requests.get(zip_url, stream=True)
total_size = int(response.headers.get("content-length", 0))
block_size = 1024  # 1KB
progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

with open("dji_thermal_sdk.zip", "wb") as zip_file:
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        zip_file.write(data)

progress_bar.close()

# Extract the contents of the zip file
with zipfile.ZipFile("dji_thermal_sdk.zip", "r") as zip_ref:
    files_to_extract = [
        file for file in zip_ref.namelist()
        if file.startswith("utility/bin/linux/release_x64") or file.startswith("utility/bin/windows/release_x64")
    ]
    progress_bar = tqdm(total=len(files_to_extract))

    for file in files_to_extract:
        zip_ref.extract(file, target_dir)
        progress_bar.update(1)

progress_bar.close()

# Remove the zip file
os.remove("dji_thermal_sdk.zip")

print("Installation completed.")
