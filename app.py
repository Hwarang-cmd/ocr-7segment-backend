from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
import re
import os

app = Flask(__name__)

def preprocess_image(image_cv):
    h, w = image_cv.shape[:2]
    x1, y1 = int(w * 0.25), int(h * 0.15)
    x2, y2 = int(w * 0.75), int(h * 0.85)
    cropped = image_cv[y1:y2, x1:x2]

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    resized = cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return resized

@app.route("/ocr", methods=["POST"])
def ocr():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    processed = preprocess_image(image_cv)

    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    text = pytesseract.image_to_string(processed, config=config)
    digits_only = re.sub(r'\D', '', text)

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
