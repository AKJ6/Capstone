from ultralytics import YOLO

# Load YOLOv8 model (Pretrained on COCO dataset)
model = YOLO("yolov8s.pt")

# Train the model
model.train(data="/home/akhilesh/Documents/Capstone/dataset/data.yaml", epochs=100, imgsz=640, batch=16, device="cuda")  # Change "cuda" to "cpu" if needed

# Save the trained model
model.export(format="onnx")  # Export the model in ONNX format for future inference
