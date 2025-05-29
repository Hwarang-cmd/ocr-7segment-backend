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

    # เพิ่ม contrast ด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # ถ้าพื้นหลังสว่าง invert ภาพ
    if np.mean(enhanced) > 127:
        enhanced = cv2.bitwise_not(enhanced)

    # Threshold แยกเลขออกจากพื้นหลัง
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # ปิดรูเชื่อม segment
    kernel = np.ones((2, 2), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # ขยายภาพเพื่อช่วย OCR อ่านง่ายขึ้น
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

    # ไม่แบ่ง ส่วน ให้ส่ง raw ทั้งหมด และ parsed เป็น raw list เดียว
    return jsonify({
        "raw": digits_only,
        "parsed": [digits_only] if digits_only else []
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
