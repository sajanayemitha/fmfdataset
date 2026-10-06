import cv2
from pathlib import Path
from datetime import datetime
import json


# -----------------------------
# Load configuration
# -----------------------------

with open("config.json", "r") as f:
    config = json.load(f)

class_names = config["class_names"]


# -----------------------------
# Select person
# -----------------------------

print("Available classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

class_index = int(input("Enter class number: "))

class_name = class_names[class_index]

save_dir = Path("raw") / class_name
save_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Haar Cascade
# -----------------------------

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    raise RuntimeError("Could not load Haar Cascade.")


# -----------------------------
# Webcam
# -----------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")


count = len(list(save_dir.glob("*.jpg")))

print()
print(f"Capturing images for: {class_name}")
print("Press SPACE to capture")
print("Press Q to quit")


while True:

    ret, frame = cap.read()

    if not ret:
        print("Could not read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )


    # -----------------------------
    # Draw detected faces
    # -----------------------------

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )


    # Counter
    cv2.putText(
        frame,
        f"Class: {class_name} | Images: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Face Capture", frame)


    key = cv2.waitKey(1) & 0xFF


    # -----------------------------
    # Capture
    # -----------------------------

    if key == ord(" "):

        if len(faces) == 0:

            print("No face detected. Image not saved.")

            continue


        # Capture first detected face
        x, y, w, h = faces[0]


        # 20% padding
        pad_x = int(0.20 * w)
        pad_y = int(0.20 * h)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)


        face = frame[y1:y2, x1:x2]


        # Filename
        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        filename = (
            f"{class_name}_{timestamp}_{count + 1}.jpg"
        )

        save_path = save_dir / filename


        cv2.imwrite(
            str(save_path),
            face
        )

        count += 1

        print(f"Saved: {save_path}")


    # -----------------------------
    # Quit
    # -----------------------------

    elif key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()