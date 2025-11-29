# 1. Gunakan base image Python yang ringan
FROM python:3.10-slim

# 2. Set folder kerja di dalam server
WORKDIR /app

# 3. Instal dependensi sistem yang dibutuhkan oleh OpenCV & InsightFace
# Library libgl1 dan libglib2 seringkali dibutuhkan agar cv2 tidak error
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy semua file dari komputer kita ke server
COPY . .

# 5. Instal library Python dari requirements.txt
# --no-cache-dir agar ukuran image tetap kecil
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Railway akan memberikan variabel environment PORT
# Kita expose port tersebut (meskipun Railway menghandle ini otomatis, ini good practice)
ENV PORT=5000

# 7. Perintah untuk menjalankan aplikasi saat server nyala
# -b 0.0.0.0:$PORT artinya listen di semua IP pada PORT yang disediakan Railway
CMD gunicorn app:app -b 0.0.0.0:$PORT