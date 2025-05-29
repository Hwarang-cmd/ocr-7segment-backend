from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import re
import os
import numpy as np
import cv2

app = Flask(__name__)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # เน้น contrast และลด noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Invert ถ้าพื้นหลังสว่างกว่าตัวเลข
    mean_val = np.mean(enhanced)
    if mean_val > 127:
        enhanced = cv2.bitwise_not(enhanced)

    # Thresholding เพื่อแยกตัวเลข
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing เพื่อเชื่อม segment ที่ขาด
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize ให้ใหญ่ขึ้นเพื่อช่วย OCR
    resized = cv2.resize(closed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)

    processed_img = preprocess_image(image_np)

    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(processed_img, config=config)

    digits_only = re.sub(r'\D', '', text)

    # ตรรกะการแบ่ง SYS, DIA, PULSE
    parts = []
    if len(digits_only) == 9:
        parts = [digits_only[0:3], digits_only[3:6], digits_only[6:9]]
    elif len(digits_only) == 8:
        parts = [digits_only[0:3], digits_only[3:5], digits_only[5:8]]
    elif len(digits_only) == 7:
        parts = [digits_only[0:3], digits_only[3:5], digits_only[5:7]]
    elif len(digits_only) == 6:
        parts = [digits_only[0:3], digits_only[3:5], digits_only[5:6]]
    elif len(digits_only) == 5:
        parts = [digits_only[0:3], digits_only[3:5], '']
    elif len(digits_only) == 4:
        parts = [digits_only[0:3], digits_only[3:4], '']
    else:
        parts = [digits_only]

    return jsonify({
        "raw": digits_only,
        "parsed": parts
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
