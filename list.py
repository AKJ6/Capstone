from ultralytics import YOLO

# Load your custom YOLO model
model = YOLO("yolov8s.pt")  # Replace with your custom model file

# Get class names
class_names = model.names  # Dictionary mapping class indices to names

# Print the class names
print("Objects this model can detect:")
for idx, name in class_names.items():
    print(f"{idx}: {name}")
