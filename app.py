from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import re
import os
import numpy as np
import cv2

app = Flask(__name__)

def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    resized = cv2.resize(closed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return resized

def detect_7segment_rois(image_np):
    """Detect ROI ที่น่าจะเป็น 7-segment display ในภาพ"""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # invert ถ้าพื้นหลังขาวและตัวเลขดำ
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # หา contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rois = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)

        # กรองสัดส่วนและขนาดที่น่าจะเป็นเลข 7-seg (กว้างกว่า สูงกว่า และสัดส่วนพอเหมาะ)
        aspect_ratio = w / float(h)
        area = w * h
        if area < 1000 or area > 20000:  # กรองขนาดเล็กเกินไปหรือใหญ่เกินไป
            continue
        if aspect_ratio < 1.5 or aspect_ratio > 5.0:  # 7-seg display มักจะกว้างกว่าสูง
            continue

        roi = image_np[y:y+h, x:x+w]
        rois.append(roi)

    # เรียง ROI จากซ้ายไปขวา (เพื่ออ่านเลขเรียงตามตำแหน่ง)
    rois = sorted(rois, key=lambda r: cv2.boundingRect(r)[0])
    return rois

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)

    rois = detect_7segment_rois(image_np)
    digits_all = ""

    for roi in rois:
        processed_roi = preprocess_roi(roi)
        config = "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(processed_roi, config=config)
        digits = re.sub(r'\D', '', text)
        digits_all += digits

    return jsonify({
        "raw": digits_all,
        "parsed": [digits_all] if digits_all else []
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
