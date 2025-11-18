from ultralytics import YOLO

# Load a model 
model = YOLO('yolov8n.pt') # Build a pre-trained model

# Train the model
model.train(data='data.yaml', epochs=100, imgsz=640, batch=16, name='experiment_1', cache=True)
