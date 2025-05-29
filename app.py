from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io

app = Flask(__name__)

def extract_digits_from_roi(image, roi):
    x, y, w, h = roi
    cropped = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    config = "--psm 7 digits"
    text = pytesseract.image_to_string(resized, config=config)
    return text.strip()

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # 🔍 กำหนดตำแหน่ง bounding boxes (x, y, w, h) สำหรับ SYS / DIA / PULSE
    # *** ต้องปรับตามตำแหน่งจริงจากภาพเครื่องของคุณ ***
    h, w = image_cv.shape[:2]
    rois = {
        "sys": (int(w*0.25), int(h*0.2), int(w*0.5), int(h*0.15)),
        "dia": (int(w*0.25), int(h*0.4), int(w*0.5), int(h*0.15)),
        "pulse": (int(w*0.25), int(h*0.6), int(w*0.5), int(h*0.15)),
    }

    results = {}
    for key, roi in rois.items():
        value = extract_digits_from_roi(image_cv, roi)
        results[key] = value

    return jsonify(results)
