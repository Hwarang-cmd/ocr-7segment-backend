FROM python:3.10-slim

# ติดตั้ง tesseract-ocr และ dependencies สำหรับ OpenCV
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libsm6 libxext6 libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอกไฟล์ requirements.txt ถ้ามี
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app"]
