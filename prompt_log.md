# Prompt Log

## Entrada 1 — 2026-09-15 — Setup inicial

Prompt inicial del proyecto: montar SOLO la estructura del sistema RAG (sin lógica todavía). Se crearon las carpetas `data/` y `src/`, el archivo `requirements.txt` con las dependencias (llama-index, chromadb, llama-index-vector-stores-chroma, llama-index-embeddings-huggingface, pypdf), este `prompt_log.md`, el `README.md` con el título del proyecto y el `.gitignore` para Python. Después se creó el entorno virtual `venv` y se instalaron las dependencias.

Nota de versiones: la combinación inicial (chromadb 0.5.23 + llama-index 0.12.x) falló en Windows porque `chromadb` 0.5.x arrastra `chroma-hnswlib`, que se compila desde C++ y requiere Microsoft Visual C++ Build Tools (ausentes). Se subió el stack a `chromadb==1.0.21` (wheels precompilados, ya sin `chroma-hnswlib`) y se alineó LlamaIndex a la línea 0.14 para mantener compatibilidad: `llama-index==0.14.24`, `llama-index-vector-stores-chroma==0.6.0`, `llama-index-embeddings-huggingface==0.8.0`, `pypdf==5.6.0`.

## Entrada 2 — 2026-09-15 — Script de ingesta

Prompt: mover `handbook.pdf` a `data/` y crear el script de ingesta. Se movió el PDF a `data/handbook.pdf` y se creó `src/ingest.py`, que lee el PDF, lo trocea (SentenceSplitter, chunk_size=512, overlap=50), genera embeddings localmente con `BAAI/bge-small-en-v1.5` (HuggingFace, sin API key) y los persiste en ChromaDB (`chroma_db/`, colección `employee_handbook`). Se desactiva el LLM (`Settings.llm = None`) porque la ingesta solo necesita embeddings.

Problemas encontrados y resueltos durante la ingesta:
1. **Caché de HuggingFace en app empaquetada (MSIX):** LlamaIndex pasa por defecto un `cache_folder` bajo `AppData\Local\llama_index\...`, que la app de escritorio de Claude virtualiza a `F:\WpSystem\...`, donde falla la escritura/symlink de `huggingface_hub` en Windows. Solución: pasar `cache_folder` explícito a `HuggingFaceEmbedding` apuntando a `.hf_cache/` (ruta normal del proyecto) + variables `HF_HUB_DISABLE_SYMLINKS`. Se añadió `.hf_cache/` al `.gitignore`.
2. **PDF no parseado:** faltaba `llama-index-readers-file`, sin el cual `SimpleDirectoryReader` leía el PDF como texto plano (bytes crudos `%PDF-1.4...`). Se añadió `llama-index-readers-file==0.7.0` a `requirements.txt` (que a su vez subió `pypdf` a 6.18.1, actualizado también en el pin). Tras reinstalar y borrar el índice contaminado, la ingesta produjo 30 chunks con texto limpio a partir de 9 páginas.

Resultado final verificado: colección `employee_handbook` con 30 vectores de dimensión 384, texto extraído correctamente.

## Entrada 3 — 2026-09-15 — Script de consulta (query.py)

Prompt: aclaración de arquitectura + creación de `src/query.py`. El handbook exige inferencia local y prohíbe enviar datos confidenciales a modelos públicos (OpenAI/ChatGPT), y la tarea permite un LLM simulado, así que NO se usa OpenAI. Requisitos del script:
- Cargar el índice de Chroma ya persistido (sin re-ingestar).
- Recibir una pregunta, recuperar top-k chunks y mostrar de qué parte del documento vienen.
- Aplicar limpieza de whitespace al texto recuperado.
- LLM intercambiable: por defecto `MockLLM` / respuesta simulada extractiva sintetizada desde los chunks, sin API key; dejar comentado cómo cambiar a un modelo local tipo Ollama.
- System prompt de "grounding": si la respuesta no está en los chunks, responder exactamente "I do not know".

Implementación en `src/query.py`: `load_index()` reconstruye el índice con `VectorStoreIndex.from_vector_store` (sin re-ingesta); `Settings.llm = MockLLM()` evita cualquier llamada a OpenAI; `get_llm()` es el punto de intercambio del LLM (devuelve None → respuesta simulada; incluye el snippet comentado de Ollama); `clean_whitespace()` colapsa el ruido `\n \n` de pypdf; `retrieve()` recupera top-k y filtra por `SIMILARITY_CUTOFF=0.35` (primera línea del grounding); si no quedan chunks se devuelve exactamente "I do not know".

Ajuste de calidad: se añadió `query_instruction` al `HuggingFaceEmbedding` de la consulta (los modelos BGE recuperan mejor con prefijo en la query; los pasajes ya embebidos no lo llevan — uso asimétrico).

Verificación: pregunta sobre confidencialidad → score 0.507 (Sección 1.2 correcta); pregunta sobre días en oficina → 0.411 ("Tuesday through Thursday"); pregunta fuera de tema (capital de Francia) → "I do not know". Nota: el PDF es un documento dummy adversarial sin sección de vacaciones; una pregunta sobre "vacaciones" recupera contenido tangencial (~0.37) porque supera el umbral aunque el tema no exista — distinguir "en dominio pero no cubierto" es tarea del LLM real vía el grounding prompt.

Observación de seguridad (pendiente): la Sección 3 del handbook está marcada "RESTRICTED ACCESS" (visible_to: admin) y contiene PII dummy (SSN, salarios). El retriever actual la devuelve sin control de acceso ni redacción de PII. El propio handbook (Sección 4.1 "Airlock") pide redactar SSN/tarjetas antes de que los datos salgan de la infraestructura privada. Queda anotado para un paso futuro (filtrado por metadatos de acceso + redacción de PII).

## Entrada 4 — 2026-09-15 — Airlock / capa de privacidad (3 capas)

Prompt: implementar la Airlock (core de la prueba) en tres capas separadas y claras.

CAPA 1 — Filtrado por rol (metadata filtering):
- Ingesta (`src/ingest.py`) reescrita: en vez de leer el PDF por páginas, se parsea con pypdf y se trocea por cabeceras "SECTION N:", creando un `Document` por sección. Cada sección se etiqueta con `access_level` = admin/public detectando el propio marcador del documento (`visible_to": "admin"` o "RESTRICTED ACCESS"), sin hardcodear "Sección 3". Los chunks heredan la metadata. La metadata se excluye del texto embebido/LLM. Resultado: 6 secciones (1 admin, 5 public) → 18 chunks (1 admin, 17 public).
- Query (`src/query.py`): `build_role_filters(user_role)` → si el rol no es admin, se aplica un `MetadataFilters` con `access_level != admin`, de modo que los chunks restringidos se excluyen en la propia consulta a Chroma. Nuevo parámetro CLI `--role` (default `employee`).

CAPA 2 — Bloqueo de entrada por PII (pre-procesamiento): `is_pii_request()` revisa la pregunta con keywords ("ssn", "social security", "credit card", "tarjeta de crédito", ...) + regex de SSN/tarjeta. Si detecta una petición de PII, `answer_question` devuelve exactamente "This request has been blocked for security reasons." ANTES de tocar el retriever o el LLM (rechazo puramente local). Requisito 3 de la prueba.

CAPA 3 — Redacción de salida (Airlock, Sección 4.1): `redact_pii()` reemplaza SSN (regex XXX-XX-XXXX) y tarjetas por "[REDACTED]". Se aplica vía `clean_and_redact()` a todo chunk recuperado antes de mostrarlo Y antes de construir el contexto del LLM, para todos los roles.

Ajuste: se subió `SIMILARITY_CUTOFF` de 0.35 a 0.40 para que, tras excluir la Sección 3, una consulta de un empleado sobre el salario del CEO no recupere coincidencias tangenciales (portada) y caiga en "I do not know".

Verificación de los 3 casos: (a) empleado pregunta salario del CEO → "I do not know" (Sección 3 no recuperada); (b) "What is the CEO's SSN?" → bloqueado en CAPA 2 sin consultar Chroma ni LLM; (c) admin pregunta por compensación → recupera la Sección 3 (access=admin) con el SSN mostrado como "[REDACTED]". Consultas legítimas siguen funcionando (confidencialidad → Sección 1; días en oficina → Sección 2, score 0.517).

## Entrada 5 — 2026-09-15 — Dockerización

Prompt: crear `Dockerfile` y `docker-compose.yml` (Python slim, instalar requirements, copiar src/ y data/, manejar bien `.hf_cache` y `chroma_db`, poder levantar con `docker compose up` y luego consultar).

Archivos creados: `Dockerfile`, `docker-compose.yml`, `docker/entrypoint.sh`, `.dockerignore`.

Decisiones de arquitectura:
- **Base:** `python:3.12-slim`. Todas las dependencias (torch, chromadb 1.x, onnxruntime, tokenizers...) tienen wheels manylinux para cp312, así que no hace falta compilador/build-essential.
- **Modelo de embeddings → BUILD time:** se instancia `HuggingFaceEmbedding(...)` en un `RUN` para hornear el modelo (~130 MB) en `/app/.hf_cache` dentro de la imagen. Así el contenedor no descarga el modelo en cada arranque y puede operar sin red. `.hf_cache` NO es un volumen (si lo fuera, el volumen vacío taparía lo horneado y forzaría re-descarga).
- **Ingesta → paso INICIAL (entrypoint), no build:** `chroma_db` es un volumen nombrado persistente; un índice horneado en build sería tapado por el volumen vacío en el primer arranque. Por eso el `entrypoint.sh` corre la ingesta de forma IDEMPOTENTE (solo si la colección está vacía) al primer `docker compose up`, escribiendo en el volumen. En arranques posteriores se omite.
- **Contenedor de larga duración:** el RAG es una CLI, no un servidor. El entrypoint termina con `tail -f /dev/null` para mantener el contenedor vivo y permitir `docker compose exec rag python src/query.py "..." --role ...`.
- **`.dockerignore`:** excluye `venv/`, `.hf_cache/`, `chroma_db/` locales para no inflar el contexto de build.

Nota: en el entorno de desarrollo actual no hay Docker instalado, así que los archivos no se pudieron construir/ejecutar aquí; se validó la coherencia de rutas (PROJECT_ROOT=/app dentro del contenedor) manualmente. Pendiente: probar `docker compose up --build` en una máquina con Docker.

## Entrada 6 — 2026-09-15 — torch CPU-only + README

Prompt: (1) aplicar la optimización de torch CPU-only para adelgazar la imagen (el RAG no usa GPU); (2) escribir el README.md completo en español.

1. torch CPU-only: en el `Dockerfile`, antes de `pip install -r requirements.txt`, se añadió `RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu`. Al quedar torch (variante CPU) ya satisfecho, el paso de requirements no vuelve a instalarlo ni baja las librerías CUDA, reduciendo la imagen en varios GB. Nota: si ese pin de versión no estuviera en el índice CPU de PyTorch en el momento del build, habría que ajustar la versión.

2. README.md reescrito (antes solo tenía el título): descripción del proyecto, stack, arquitectura (ingesta por secciones + etiquetado de acceso, retrieval con grounding, las tres capas de la Airlock), estructura, instrucciones de Docker (`docker compose up --build`), decisiones de empaquetado y los comandos de ejemplo de los tres casos de seguridad.
