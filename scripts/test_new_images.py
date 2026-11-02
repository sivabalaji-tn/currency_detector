from ultralytics import YOLO


def main():
    model = YOLO(
        r"M:\currency-detector\runs\train\rupee50_yolov8n\weights\best.pt"
    )

    model.predict(
        source=r"M:\currency-detector\temp",
        imgsz=640,
        conf=0.25,
        device=0,
        save=True,
        show=False,
        project=r"M:\currency-detector\runs\detect",
        name="new_image_test"
    )

    print("\nNew image testing completed!")


if __name__ == "__main__":
    main()