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

    # เพิ่ม contrast และลด noise
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Invert ถ้าพื้นหลังสว่าง
    if np.mean(enhanced) > 127:
        enhanced = cv2.bitwise_not(enhanced)

    # Threshold เพื่อแยกตัวเลข
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ปิดรูเพื่อเชื่อม segment
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize ขยายภาพเพื่อ OCR
    resized = cv2.resize(closed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

def parse_digits(digits):
    length = len(digits)
    if length == 9:
        return [digits[0:3], digits[3:6], digits[6:9]]
    elif length == 8:
        return [digits[0:3], digits[3:5], digits[5:8]]
    elif length == 7:
        return [digits[0:3], digits[3:5], digits[5:7]]
    elif length == 6:
        return [digits[0:3], digits[3:5], digits[5:6]]
    elif length == 5:
        return [digits[0:3], digits[3:5], '']
    elif length == 4:
        return [digits[0:3], digits[3:4], '']
    else:
        return [digits]

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    try:
        image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {str(e)}"}), 400

    image_np = np.array(image_pil)
    processed_img = preprocess_image(image_np)

    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(processed_img, config=config)

    digits_only = re.sub(r'\D', '', text)

    parsed = parse_digits(digits_only)

    return jsonify({
        "raw": digits_only,
        "parsed": parsed
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
