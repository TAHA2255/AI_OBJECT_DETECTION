from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO("yolov8n.pt")

# Run inference  with arguments
results = model(source="cricket.mp4", show=True, conf=0.5, save=True)