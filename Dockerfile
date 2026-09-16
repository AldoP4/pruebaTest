# --- Imagen base: Python slim (Debian) ---
FROM python:3.12-slim

# Variables de entorno:
# - PYTHONUNBUFFERED: logs en tiempo real (sin buffer).
# - PYTHONDONTWRITEBYTECODE: no generar .pyc.
# - HF_HOME / HF_HUB_DISABLE_SYMLINKS: la caché de HuggingFace vive en
#   /app/.hf_cache y se evita el uso de symlinks (igual que en local). En Linux
#   los symlinks funcionan, pero desactivarlos hace el comportamiento
#   determinista y consistente con el entorno de desarrollo.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf_cache \
    HF_HUB_DISABLE_SYMLINKS=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

# --- Dependencias de Python ---
# Se copia primero requirements.txt para aprovechar la caché de capas de Docker:
# si el código cambia pero las dependencias no, no se reinstala todo.
COPY requirements.txt .

# torch CPU-only: el RAG no usa GPU. Instalamos torch desde el índice CPU de
# PyTorch ANTES que el resto, para evitar las librerías CUDA (que añaden varios
# GB a la imagen). Al quedar torch ya satisfecho, `pip install -r requirements`
# no vuelve a tocarlo ni baja la variante GPU.
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# --- Código y datos ---
COPY src/ ./src/
COPY data/ ./data/

# --- Pre-descarga del modelo de embeddings EN BUILD TIME ---
# Instanciamos el modelo una vez durante el build para que quede cacheado en
# /app/.hf_cache dentro de la imagen. Así el contenedor NO descarga ~130 MB en
# cada arranque y puede operar sin red después. Se usa el mismo cache_folder que
# los scripts, para que reutilicen exactamente esta caché.
RUN python -c "from llama_index.embeddings.huggingface import HuggingFaceEmbedding; \
HuggingFaceEmbedding(model_name='BAAI/bge-small-en-v1.5', cache_folder='/app/.hf_cache')"

# --- Script de arranque ---
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
