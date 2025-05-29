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
    # Threshold แบบ adaptive และ invert เพราะเลขเป็นสีดำบนพื้นสีเขียวสว่าง
    thresh = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    resized = cv2.resize(closed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    return resized

def mask_green_background(image_np):
    # แปลงเป็น HSV เพื่อแยกสีเขียว
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # invert mask เพื่อให้ตัวเลขดำเป็นขาวใน mask
    mask_inv = cv2.bitwise_not(mask)
    return mask_inv

def detect_7segment_rois(image_np):
    # ลบพื้นหลังสีเขียวออกโดย mask
    mask_inv = mask_green_background(image_np)
    
    # เอา mask_inv มา apply กับ grayscale เพื่อแยกเลขออกจากพื้นหลัง
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    fg = cv2.bitwise_and(gray, gray, mask=mask_inv)
    
    # ทำ threshold เพื่อหา contour ตัวเลข
    _, thresh = cv2.threshold(fg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # morphology ปิดรู
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rois = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = w * h
        
        # ปรับ filter ตามลักษณะ 7 segment ในภาพนี้
        if area < 500 or area > 20000:
            continue
        if aspect_ratio < 1.5 or aspect_ratio > 5.5:
            continue
        if h < 20 or h > 80:
            continue

        roi = image_np[y:y+h, x:x+w]
        rois.append((x, roi))
    
    # เรียงจากซ้ายไปขวา
    rois = sorted(rois, key=lambda x: x[0])
    rois_only = [r[1] for r in rois]
    return rois_only

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
