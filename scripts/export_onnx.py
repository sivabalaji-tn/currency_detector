
from ultralytics import YOLO

model = YOLO(
    r"M:\currency-detector\runs\train\rupee50_yolov8n\weights\best.pt"
)

model.export(
    format="onnx",
    imgsz=640,
    opset=12,
    simplify=True
)

print("ONNX export completed!")
