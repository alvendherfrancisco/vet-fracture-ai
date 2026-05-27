# HuggingFace Spaces requires port 7860
FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user (HuggingFace requirement)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR /home/user/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY --chown=user main.py .
COPY --chown=user index.html .
COPY --chown=user style.css .
COPY --chown=user script.js .

# HuggingFace Spaces listens on 7860
EXPOSE 7860

CMD ["python", "main.py"]
