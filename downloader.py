import os
import subprocess

# Define base directory
BASE_DIR = "images"

# Read class names from file
CLASS_FILE = "classes.txt"

# Check if class file exists
if not os.path.isfile(CLASS_FILE):
    print(f"Error: File '{CLASS_FILE}' not found!")
    exit(1)

# Read class names from the file
with open(CLASS_FILE, "r") as file:
    class_names = [line.strip() for line in file if line.strip()]

# Loop through each class
for class_name in class_names:
    # Define directories
    class_dir = os.path.join(BASE_DIR, class_name)
    csv_dir = os.path.join(class_dir, "csv")

    # Create directories if they don't exist
    os.makedirs(class_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # Construct the command
    command = f'oi_download_images --base_dir "{class_dir}" --labels "{class_name}" --csv_dir "{csv_dir}"'

    print(f"Downloading images for class: {class_name}")

    # Execute the command in Bash
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error downloading images for {class_name}: {e}")

print("Download process completed.")

