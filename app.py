from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import os
import numpy as np
import cv2
import re

app = Flask(__name__)

def preprocess_image(image):
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # เบลอเพื่อลด noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold แบบ Inverse (ตัวเลขดำ พื้นขาว)
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    return binary

def find_segments(binary_image):
    # หาขอบเขตวัตถุ (contours)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digits = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # กรองเฉพาะที่น่าจะเป็นตัวเลข (ขนาดประมาณ 7 segment)
        if h > 20 and w > 10 and h < 150 and w < 100:
            digits.append((x, y, w, h))

    # เรียงจากซ้ายไปขวา
    digits = sorted(digits, key=lambda x: x[0])
    return digits

def recognize_digits(image, digits):
    result = ""
    for (x, y, w, h) in digits:
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        digit = pytesseract.image_to_string(roi, config=config)
        result += digit.strip()

    # กรองเฉพาะตัวเลข
    result = re.sub(r'\D', '', result)
    return result

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image_pil)

    binary = preprocess_image(image_np)
    digits = find_segments(binary)
    raw_digits = recognize_digits(binary, digits)

    return jsonify({
        "raw": raw_digits,
        "parsed": [raw_digits]  # ยังไม่แบ่งในขั้นตอนนี้
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
