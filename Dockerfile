# 1. Gunakan base image Python yang ringan
FROM python:3.10-slim

# 2. Set folder kerja di dalam server
WORKDIR /app

# 3. Instal dependensi sistem (termasuk build-essential dan libgl1)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy semua file dari komputer ke server
COPY . .

# 5. Instal library Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- PERBAIKAN UTAMA DI SINI ---
# 6. Pre-download model InsightFace 'buffalo_l' agar tidak download saat runtime
# Kita buat folder default insightface dan download manual
RUN mkdir -p /root/.insightface/models/buffalo_l && \
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/buffalo_l.zip && \
    unzip /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip

# 7. Expose port
ENV PORT=5000

# 8. Jalankan aplikasi
CMD gunicorn app:app -b 0.0.0.0:$PORT