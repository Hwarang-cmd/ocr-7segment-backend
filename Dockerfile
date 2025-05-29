FROM python:3.10-slim

# ติดตั้ง tesseract-ocr และ dependencies สำหรับ OpenCV, รวม build tools เผื่อจำเป็น
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libsm6 libxext6 libxrender-dev \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# คัดลอก requirements.txt แล้วติดตั้ง dependencies python
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดแอปทั้งหมด
COPY . .

# รันแอปด้วย gunicorn บนพอร์ต 80
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app"]
