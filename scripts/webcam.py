
import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime


MODEL_PATH = r"M:\currency-detector\runs\train\rupee50_yolov8n\weights\best.pt"

def main():
    print("Loading ₹50 Currency Detection Model...")

    model = YOLO(MODEL_PATH)

    # Open laptop webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        print("Try changing camera index 0 to 1.")
        return

    print("\nWebcam started successfully!")
    print("Press Q to quit.")
    print("Press S to save a screenshot.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Unable to read camera frame.")
            break

        # Run YOLO detection using RTX 3050
        results = model.predict(
            source=frame,
            imgsz=640,
            conf=0.35,
            device=0,
            verbose=False
        )

        result = results[0]
        output = result.plot()

        # Detection status
        detected = len(result.boxes) > 0

        if detected:
            status = "50 RUPEE NOTE DETECTED"
            color = (0, 255, 0)
        else:
            status = "50 RUPEE NOT DETECTED"
            color = (0, 0, 255)

        # Display instructions
        cv2.rectangle(output, (5, 5), (output.shape[1] - 5, 105),
                      (25, 25, 25), -1)

        cv2.putText(
            output, "INDIAN CURRENCY DETECTOR",
            (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        cv2.putText(
            output, "Show a complete Rs.50 note to the camera",
            (15, 55), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        cv2.putText(
            output, "Keep note visible | Use good lighting",
            (15, 78), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 1
        )

        cv2.putText(
            output, "Q: Quit   S: Screenshot",
            (15, 98), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 255, 255), 1
        )

        # Display detection result
        cv2.putText(
            output, status,
            (15, output.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2
        )

        cv2.imshow("Indian Rs.50 Currency Detector", output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):
            save_dir = Path(r"M:\currency-detector\runs\detect\webcam")
            save_dir.mkdir(parents=True, exist_ok=True)

            filename = datetime.now().strftime(
                "capture_%Y%m%d_%H%M%S.jpg"
            )

            cv2.imwrite(str(save_dir / filename), output)
            print(f"Screenshot saved: {filename}")

    camera.release()
    cv2.destroyAllWindows()

    print("\nWebcam detection stopped.")


if __name__ == "__main__":
    main()
