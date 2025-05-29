from flask import Flask, request, jsonify
from PIL import Image
import easyocr
import io
import numpy as np
import cv2
import re
import os
import requests

app = Flask(__name__)
reader = easyocr.Reader(['en'], gpu=False)  # ตั้งค่า gpu=True ถ้ามี GPU

def preprocess_image(image_np):
    # แปลงเป็น grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # ปรับ contrast ด้วย CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Threshold แบบ Otsu
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert ถ้าพื้นหลังสว่าง (ให้ตัวเลขเป็นขาว)
    if np.mean(thresh) > 127:
        thresh = cv2.bitwise_not(thresh)

    # ขยายภาพ (resize) เพื่อช่วย OCR
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return resized

@app.route("/ocr", methods=["POST"])
def ocr():
    # รับไฟล์ภาพ
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_np = np.array(image_pil)

    # preprocess
    proc_img = preprocess_image(image_np)

    # OCR ด้วย EasyOCR
    result = reader.readtext(proc_img)

    # รวมเฉพาะตัวเลขที่ detect ได้
    digits = ''.join([res[1] for res in result if re.fullmatch(r'\d+', res[1])])

    return jsonify({
        "raw": digits,
        "parsed": digits  # ยังไม่แยกบรรทัดตามที่ต้องการ
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
