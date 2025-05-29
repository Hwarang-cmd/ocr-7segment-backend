from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
import io
import re
import os
import numpy as np
import cv2

app = Flask(__name__)

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ถ้าต้องการแปลงเป็น numpy array เพื่อ preprocess เบื้องต้น
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # (Optional) ทำ threshold หรือปรับภาพให้ง่ายต่อ OCR
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # แปลงกลับเป็น PIL Image ก่อนส่งให้ pytesseract
    processed_img = Image.fromarray(thresh)

    # เรียก Tesseract OCR โดยระบุ whitelist เฉพาะตัวเลข
    config = "--psm 6 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(processed_img, config=config)

    # ลบตัวที่ไม่ใช่เลขออก
    digits_only = re.sub(r'\D', '', text)

    # แบ่งส่วนตามความยาวเลขที่อ่านได้
    if len(digits_only) == 9:
        parts = [digits_only[0:3], digits_only[3:6], digits_only[6:9]]
    elif len(digits_only) == 8:
        parts = [digits_only[0:3], digits_only[3:5], digits_only[5:8]]
    elif len(digits_only) == 7:
        parts = [digits_only[0:3], digits_only[3:5], digits_only[5:7]]
    elif len(digits_only) == 6:
        parts = [digits_only[0:3], digits_only[3:5], digits_only[5:6]]
    else:
        parts = [digits_only]

    return jsonify({
        "raw": digits_only,
        "parsed": parts
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
