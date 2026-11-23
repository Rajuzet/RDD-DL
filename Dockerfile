# Production Multi-Stage Dockerfile for Retinal OCT Diagnostic Service
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specifications first to leverage Docker layer caching
COPY requirements.txt setup.py ./

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Copy source repository
COPY . .

# Generate sample clinical dataset and checkpoints if not already populated
RUN python scripts/generate_sample_data.py

EXPOSE 8501 8000

# Default entrypoint launches Streamlit studio
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
