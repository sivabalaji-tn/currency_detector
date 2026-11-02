
import cv2
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime


def main():

    parser = argparse.ArgumentParser(
        description="Indian ₹50 Currency Detector using ONNX"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model"
    )

    args = parser.parse_args()

    MODEL_PATH = Path(args.model)

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    print("Loading ₹50 Currency Detection ONNX Model...")

    # Load ONNX model directly using OpenCV
    net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    print("ONNX Model loaded successfully!")

    # Open laptop webcam
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        print("Try changing camera index 0 to 1.")
        return

    print("\nWebcam started successfully!")
    print("Press Q to quit.")
    print("Press S to save a screenshot.")

    confidence_threshold = 0.35
    input_size = 640

    while True:

        success, frame = camera.read()

        if not success:
            print("Unable to read camera frame.")
            break

        height, width = frame.shape[:2]

        # Prepare image for ONNX model
        blob = cv2.dnn.blobFromImage(
            frame,
            scalefactor=1 / 255.0,
            size=(input_size, input_size),
            swapRB=True,
            crop=False
        )

        net.setInput(blob)
        predictions = net.forward()

        # YOLOv8 output: (1, 5, 8400)
        predictions = predictions[0].T

        boxes = []
        confidences = []

        scale_x = width / input_size
        scale_y = height / input_size

        for detection in predictions:

            confidence = float(detection[4])

            if confidence < confidence_threshold:
                continue

            center_x = float(detection[0]) * scale_x
            center_y = float(detection[1]) * scale_y

            box_width = float(detection[2]) * scale_x
            box_height = float(detection[3]) * scale_y

            x = int(center_x - box_width / 2)
            y = int(center_y - box_height / 2)

            boxes.append([
                x, y,
                int(box_width),
                int(box_height)
            ])

            confidences.append(confidence)

        # Remove duplicate detections
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            confidence_threshold,
            0.45
        )

        detected = len(indices) > 0

        # Draw bounding boxes
        if detected:

            for index in np.array(indices).flatten():

                x, y, w, h = boxes[index]
                confidence = confidences[index]

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"50_rupee {confidence:.2f}",
                    (x, max(y - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            status = "50 RUPEE NOTE DETECTED"
            color = (0, 255, 0)

        else:

            status = "50 RUPEE NOT DETECTED"
            color = (0, 0, 255)

        # Display instructions
        cv2.rectangle(
            frame,
            (5, 5),
            (width - 5, 105),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            frame,
            "INDIAN CURRENCY DETECTOR",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Show a complete Rs.50 note to the camera",
            (15, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Keep note visible | Use good lighting",
            (15, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            "Q: Quit   S: Screenshot",
            (15, 98),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        # Display detection status
        cv2.putText(
            frame,
            status,
            (15, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.imshow(
            "Indian Rs.50 Currency Detector - ONNX",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("s"):

            save_dir = Path(
                "M:/currency-detector/runs/detect/onnx_webcam"
            )

            save_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            filename = datetime.now().strftime(
                "capture_%Y%m%d_%H%M%S.jpg"
            )

            cv2.imwrite(
                str(save_dir / filename),
                frame
            )

            print(f"Screenshot saved: {filename}")

    camera.release()
    cv2.destroyAllWindows()

    print("\nWebcam detection stopped!")


if __name__ == "__main__":
    main()
