FROM python:3.11-slim

# System deps for MediaPipe / OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps from requirements.txt (pip-based, no uv needed)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY pretrained/ ./pretrained/

# Download MediaPipe model if not bundled
RUN python -c "
import urllib.request, os
os.makedirs('pretrained', exist_ok=True)
model = 'pretrained/hand_landmarker.task'
if not os.path.exists(model):
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        model
    )
    print('Downloaded MediaPipe model')
else:
    print('MediaPipe model already present')
" || echo "MediaPipe model download skipped (will retry at runtime)"

EXPOSE 5000

# Use gunicorn with gthread workers (required for SSE streaming)
CMD ["gunicorn", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "120", \
     "--worker-class", "gthread", \
     "--chdir", "src", \
     "app:app"]
