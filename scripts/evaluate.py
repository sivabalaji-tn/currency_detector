
from ultralytics import YOLO


def main():
    model = YOLO(
        r"M:\currency-detector\runs\train\rupee50_yolov8n\weights\best.pt"
    )

    metrics = model.val(
        data=r"M:\currency-detector\dataset\data.yaml",
        split="test",
        imgsz=640,
        batch=4,
        device=0,
        workers=0
    )

    print("\nTest evaluation completed!")


if __name__ == "__main__":
    main()
