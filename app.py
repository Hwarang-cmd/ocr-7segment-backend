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

    # ใช้ CLAHE เพื่อเพิ่ม contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # ใช้ adaptive threshold (invert เพราะเลขดำบนพื้นเขียว)
    thresh = cv2.adaptiveThreshold(enhanced, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 5)

    # Morphology: ปิดรูเล็ก ๆ เพื่อเชื่อมตัวเลข 7-segment
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # ขยายภาพเพื่อช่วย OCR อ่านง่ายขึ้น
    resized = cv2.resize(closed, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
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

    # config Tesseract: PSM 7 (single line), whitelist เลข 0-9
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(processed_img, config=config)

    digits_only = re.sub(r'\D', '', text)

    return jsonify({
        "raw": digits_only,
        "parsed": [digits_only] if digits_only else []
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
