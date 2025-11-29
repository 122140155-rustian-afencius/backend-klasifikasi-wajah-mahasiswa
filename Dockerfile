FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.insightface/models/buffalo_l && \
    wget https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip -O /tmp/buffalo_l.zip && \
    unzip /tmp/buffalo_l.zip -d /root/.insightface/models/buffalo_l && \
    rm /tmp/buffalo_l.zip

ENV PORT=5000

CMD gunicorn app:app --workers 1 --threads 8 --timeout 0 -b 0.0.0.0:$PORT