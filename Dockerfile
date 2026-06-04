# Image backend LSF — inférence uniquement (légère, CPU).
FROM python:3.11-slim

# Bonnes pratiques runtime Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# PyTorch CPU depuis l'index dédié (évite de tirer les paquets CUDA lourds).
COPY requirements-server.txt .
RUN pip install --upgrade pip && \
    pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-server.txt

# Code applicatif (le dataset src/data/approved est exclu via .dockerignore)
COPY src ./src
COPY webapp ./webapp
COPY models/lsf_v1.pt models/label_map.json ./models/

EXPOSE 8000

# Coolify mappe son proxy sur ce port. 1 worker suffit (modèle chargé en mémoire).
CMD ["uvicorn", "webapp.server:app", "--host", "0.0.0.0", "--port", "8000"]
