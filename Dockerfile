FROM python:3.13-slim

# System deps for OpenCV + pyttsx3
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        espeak \
        espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache)
COPY pyproject.toml uv.lock ./

# Install all Python deps
RUN uv sync --no-dev --frozen

# Copy application source
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/
COPY pretrained/ ./pretrained/

EXPOSE 5000

# Run with the venv created by uv
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "src/app.py"]
