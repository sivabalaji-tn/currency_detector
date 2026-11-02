
from ultralytics import YOLO


def main():
    # Load pretrained YOLOv8 Nano model
    model = YOLO("yolov8n.pt")

    # Start training
    model.train(
        data=r"M:\currency-detector\dataset\data.yaml",
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=2,
        project=r"M:\currency-detector\runs\train",
        name="rupee50_yolov8n",
        patience=20,
        amp=True
    )


if __name__ == "__main__":
    main()
