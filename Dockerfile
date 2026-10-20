FROM python:3.10-slim

WORKDIR /app

# Install minimal build tools for uroman & audio packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install CPU PyTorch using extra-index-url so dependencies resolve properly
RUN pip install --no-cache-dir torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose standard ports (Render uses 10000, Hugging Face uses 7860)
EXPOSE 10000 7860

CMD ["python", "app.py"]


