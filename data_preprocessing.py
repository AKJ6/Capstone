import os

# 🔹 Configure paths and class mappings
LABELS_DIR = "/path/to/labels"  # Change to your labels folder
CLASS_MAPPING = {0: 5}  # Change old class (key) to new class (value)

def update_labels(file_path):
    """Reads a YOLO label file, updates class IDs, and saves changes."""
    with open(file_path, "r") as file:
        lines = file.readlines()

    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:  # Skip empty lines
            continue

        try:
            class_id = int(parts[0])
            if class_id in CLASS_MAPPING:
                parts[0] = str(CLASS_MAPPING[class_id])  # Change class ID
        except ValueError:
            pass  # Skip lines that don't have a valid class ID

        updated_lines.append(" ".join(parts))

    # Only write back if changes were made
    if updated_lines:
        with open(file_path, "w") as file:
            file.write("\n".join(updated_lines) + "\n")  # Save changes

# 🔹 Process all label files
for filename in os.listdir(LABELS_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(LABELS_DIR, filename)
        
        # Skip empty files
        if os.path.getsize(file_path) == 0:
            continue
        
        update_labels(file_path)

print("✅ Labels updated successfully!")
