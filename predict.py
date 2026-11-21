from ultralytics import YOLO
import cv2
from datetime import datetime

# Load pretrained animal detection model
model = YOLO("yolov8n.pt")

# Supported animal classes in the COCO pretrained model
ANIMAL_CLASSES = [
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe"
]


def detect_animals(image, source="Image", frame_number=None, video_time=None):
    """
    Detect supported animals in an RGB image.
    Returns an annotated RGB image and detection records.
    """

    results = model.predict(
        source=image,
        conf=0.25,
        verbose=False
    )

    result = results[0]

    # YOLO draws boxes in BGR format
    annotated_image = result.plot()
    annotated_image = cv2.cvtColor(
        annotated_image, cv2.COLOR_BGR2RGB
    )

    detections = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for box in result.boxes:
        class_id = int(box.cls[0])
        animal_name = model.names[class_id]

        if animal_name in ANIMAL_CLASSES:
            confidence = float(box.conf[0])

            record = {
                "Animal": animal_name.capitalize(),
                "Confidence (%)": round(confidence * 100, 2),
                "Date & Time": now,
                "Source": source,
                "Frame": frame_number if frame_number is not None else "-",
                "Video Time (sec)": (
                    round(video_time, 2)
                    if video_time is not None else "-"
                )
            }

            detections.append(record)

    return annotated_image, detections