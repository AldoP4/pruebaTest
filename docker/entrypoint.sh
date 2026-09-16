#!/bin/sh
# Punto de entrada del contenedor.
#
# 1) Ingesta IDEMPOTENTE: solo se ejecuta si el índice de Chroma está vacío o no
#    existe todavía. Como chroma_db es un volumen persistente, la ingesta corre
#    una única vez (primer `docker compose up`) y en arranques posteriores se
#    omite.
# 2) Mantiene el contenedor vivo para poder lanzar consultas con
#    `docker compose exec`.
set -eu

# Nº de chunks ya presentes en la colección (0 si no existe).
COUNT=$(python - <<'PY'
import chromadb
try:
    client = chromadb.PersistentClient(path="/app/chroma_db")
    print(client.get_collection("employee_handbook").count())
except Exception:
    print(0)
PY
)

if [ "$COUNT" -gt 0 ]; then
    echo "[entrypoint] Índice ya presente ($COUNT chunks): se omite la ingesta."
else
    echo "[entrypoint] Índice ausente/vacío: ejecutando ingesta inicial..."
    python src/ingest.py
fi

echo ""
echo "[entrypoint] Sistema RAG listo. Para hacer consultas:"
echo "    docker compose exec rag python src/query.py \"your question\" --role employee"
echo "    docker compose exec rag python src/query.py \"...\" --role admin"
echo ""

# Mantener el contenedor en ejecución para permitir 'exec' (el RAG es una CLI,
# no un servidor de larga duración).
exec tail -f /dev/null
