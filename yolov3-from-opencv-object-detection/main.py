import os
import cv2
import numpy as np
import easyocr
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse

# ────────────────────────────────────────────────
#  Assuming you still use your util.py for YOLO helpers
#  If not → replace get_outputs() and NMS() with cv2.dnn functions
# ────────────────────────────────────────────────
import util   # your original util.py with get_outputs() and NMS()

# ────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
MODEL_CFG     = BASE_DIR / "model" / "cfg" / "darknet-yolov3.cfg"
MODEL_WEIGHTS = BASE_DIR / "model" / "weights" / "model.weights"
CLASS_NAMES   = BASE_DIR / "model" / "class.names"

OUTPUT_CSV    = BASE_DIR / "output" / "detected_plates.csv"
OUTPUT_DIR    = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CONF_THRESHOLD    = 0.40
NMS_THRESHOLD     = 0.45
OCR_CONF_THRESHOLD = 0.50

# ────────────────────────────────────────────────
# GLOBAL STATE
# ────────────────────────────────────────────────

results = []                    # list of dicts → will become CSV
reader = None
net = None

# ────────────────────────────────────────────────
# INIT
# ────────────────────────────────────────────────

def init_model():
    global net, reader

    print("Loading YOLOv3 model...")
    net = cv2.dnn.readNetFromDarknet(str(MODEL_CFG), str(MODEL_WEIGHTS))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)     # change to DNN_TARGET_CUDA if you have GPU

    print("Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    print("Model & OCR ready.\n")


def clean_plate_text(text: str) -> str:
    """Remove unwanted characters, keep alphanumeric only, upper case"""
    return ''.join(c for c in text.upper() if c.isalnum())


def process_frame(frame, save_image=False, img_name=None):
    global results

    if frame is None:
        return frame

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = util.get_outputs(net)                    # from your util.py

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            conf = scores[class_id]

            if conf > CONF_THRESHOLD:
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                width    = int(detection[2] * w)
                height   = int(detection[3] * h)

                x = center_x - width // 2
                y = center_y - height // 2

                boxes.append([x, y, width, height])
                confidences.append(float(conf))
                class_ids.append(class_id)

    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    frame_copy = frame.copy()

    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(x + w, w), min(y + h, h)

        # Draw box & confidence
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{confidences[i]:.2f}"
        cv2.putText(frame_copy, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop plate
        plate = frame[y1:y2, x1:x2]
        if plate.size == 0:
            continue

        # OCR
        ocr_results = reader.readtext(plate, detail=1, paragraph=False)
        for (_, text, ocr_conf) in ocr_results:
            if ocr_conf >= OCR_CONF_THRESHOLD:
                clean_text = clean_plate_text(text)
                if len(clean_text) < 4:
                    continue

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                entry = {
                    "timestamp": timestamp,
                    "source": img_name or "webcam",
                    "plate": clean_text,
                    "yolo_conf": round(confidences[i], 3),
                    "ocr_conf": round(ocr_conf, 3)
                }

                results.append(entry)
                print(f"[{timestamp}]  {clean_text:10}   (YOLO: {entry['yolo_conf']:.3f} | OCR: {entry['ocr_conf']:.3f})")

                # Optional: save individual crop
                if save_image:
                    cv2.imwrite(str(OUTPUT_DIR / f"plate_{timestamp.replace(':', '-')}.jpg"), plate)

    return frame_copy


def save_results():
    if not results:
        print("\nNo plates detected.")
        return

    df = pd.DataFrame(results)
    # Sort by time (useful if mixing images + webcam)
    df = df.sort_values("timestamp")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} detections → {OUTPUT_CSV}")
    print(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", action="store_true", help="Process all images in ../data")
    args = parser.parse_args()

    init_model()

    if args.image:
        # Single image mode
        img = cv2.imread(args.image)
        if img is None:
            print("Cannot read image")
            return

        processed = process_frame(img, save_image=True, img_name=Path(args.image).name)
        cv2.imwrite(str(OUTPUT_DIR / f"annotated_{Path(args.image).name}"), processed)
        cv2.imshow("Result", processed)
        cv2.waitKey(0)

    elif args.folder:
        # Batch folder mode (your original data folder)
        for p in (BASE_DIR.parent / "data").glob("*.[jpJP][pnPN][gG]*"):
            img = cv2.imread(str(p))
            if img is None:
                continue
            process_frame(img, save_image=True, img_name=p.name)

    else:
        # ── Default: Webcam live ──────────────────────────────────────
        cap = cv2.VideoCapture(0)   # 0 = default webcam

        if not cap.isOpened():
            print("Cannot open webcam")
            return

        print("Webcam opened. Press 'q' to quit and save results.\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = process_frame(frame)
            cv2.imshow("ANPR - Webcam", processed)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Final save
    save_results()


if __name__ == "__main__":
    main()
