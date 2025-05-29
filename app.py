from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import re
import os
import numpy as np
import cv2

app = Flask(__name__)

def preprocess_line(image, y_start_ratio, y_end_ratio):
    h, w = image.shape[:2]
    y1 = int(h * y_start_ratio)
    y2 = int(h * y_end_ratio)
    line_img = image[y1:y2, :]

    gray = cv2.cvtColor(line_img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # เพิ่มความคมชัดและขยายภาพ
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    enlarged = cv2.resize(cleaned, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    return enlarged

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)

    # แยกเป็น 3 โซน: บรรทัดบน กลาง ล่าง
    lines = [
        preprocess_line(image_np, 0.10, 0.33),
        preprocess_line(image_np, 0.34, 0.66),
        preprocess_line(image_np, 0.67, 0.95)
    ]

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    parsed = []
    raw_concat = ""

    for i, line in enumerate(lines):
        text = pytesseract.image_to_string(line, config=config)
        digits = re.sub(r'\D', '', text)
        parsed.append(digits)
        raw_concat += digits

    return jsonify({
        "raw": raw_concat,
        "parsed": parsed
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
