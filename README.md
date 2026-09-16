# RAG Employee Handbook

Sistema **RAG** (Retrieval-Augmented Generation) sobre un manual de empleado en PDF,
con una **capa de privacidad ("Airlock")** que aplica control de acceso por rol,
bloqueo de datos sensibles en la entrada y redacción de PII en la salida.

Diseñado para **inferencia 100% local**: los embeddings se generan en la propia
máquina y **no se envía ningún dato a modelos públicos** (OpenAI/ChatGPT), tal y
como exige el propio handbook.

---

## ¿Qué hace?

1. **Ingesta** el PDF (`data/handbook.pdf`), lo trocea, genera embeddings locales
   y los persiste en una base de datos vectorial (ChromaDB).
2. Ante una **pregunta**, recupera los fragmentos más relevantes y sintetiza una
   respuesta **basada exclusivamente en el documento** (grounding).
3. Protege la información sensible mediante las **tres capas de la Airlock**.

---

## Stack

| Componente | Tecnología |
|---|---|
| Orquestación RAG | LlamaIndex 0.14 |
| Base vectorial | ChromaDB 1.x (persistente en disco) |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace, local, 384 dims) |
| Lectura de PDF | pypdf |
| LLM de síntesis | MockLLM por defecto (sin API key); intercambiable a Ollama |
| Empaquetado | Docker + Docker Compose |

---

## Arquitectura

### 1. Ingesta — `src/ingest.py`

- El PDF se parsea con **pypdf** y se trocea **por secciones** (`SECTION N:`),
  creando un documento por sección.
- Se limpia el *whitespace* (pypdf, con PDFs de Google Docs, separa palabras con
  ruido tipo `\n \n`).
- **Etiquetado de acceso (CAPA 1):** cada sección se marca con
  `access_level = admin | public`, detectando el marcador que el propio documento
  trae (`"visible_to": "admin"` / `RESTRICTED ACCESS`). No se hardcodea "Sección 3".
- Los fragmentos se embeben con el modelo local y se persisten en `chroma_db/`
  (colección `employee_handbook`). El LLM se desactiva: la ingesta solo necesita
  embeddings.

### 2. Retrieval — `src/query.py`

- Carga el índice ya persistido (no re-ingesta).
- Recupera los **top-k** fragmentos por similitud de coseno y filtra por un
  **umbral** (`SIMILARITY_CUTOFF = 0.40`).
- **Grounding:** si tras el filtrado no queda contexto relevante, responde
  exactamente **`I do not know`** en lugar de inventar.
- **LLM intercambiable:** por defecto una respuesta simulada extractiva a partir
  de los fragmentos (sin API key). El punto de intercambio es `get_llm()`, que
  incluye comentado cómo pasar a un modelo local con **Ollama**.

### 3. Airlock — capa de privacidad (tres capas separadas)

| Capa | Momento | Función |
|---|---|---|
| **1. Filtrado por rol** | En el retrieval | Un rol no-admin aplica un filtro de metadata `access_level != admin`, excluyendo los fragmentos restringidos **en la propia consulta a Chroma**. Un empleado que pregunte por el salario del CEO no recupera la Sección 3 y cae en `I do not know`. |
| **2. Bloqueo de entrada por PII** | Antes de retrieval/LLM | Si la pregunta pide SSN o tarjetas (keywords + regex), se rechaza **localmente**, sin consultar la base vectorial ni el LLM, devolviendo `This request has been blocked for security reasons.` |
| **3. Redacción de salida** | Sobre los fragmentos recuperados | Última malla: cualquier SSN (`XXX-XX-XXXX`) o número de tarjeta se reemplaza por `[REDACTED]` antes de mostrarse o pasarse al LLM, para todos los roles. |

---

## Estructura del proyecto

```
.
├── data/handbook.pdf        # Documento fuente
├── src/
│   ├── ingest.py            # Ingesta + etiquetado de acceso (CAPA 1)
│   └── query.py             # Retrieval + grounding + Airlock (CAPAS 1/2/3)
├── docker/entrypoint.sh     # Ingesta idempotente al arrancar + mantener vivo
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── prompt_log.md            # Bitácora de prompts y decisiones
```

Rutas generadas (no versionadas): `chroma_db/` (índice) y `.hf_cache/` (modelo).

---

## Cómo levantarlo con Docker

Requiere Docker con Docker Compose. Construir y arrancar en segundo plano:

```bash
docker compose up --build -d
```

En el primer arranque, el contenedor ejecuta la **ingesta inicial** (una sola vez;
el índice queda en un volumen persistente). Para seguir el progreso:

```bash
docker compose logs -f rag
```

Detener conservando el índice:

```bash
docker compose down
```

Detener y borrar el volumen (fuerza re-ingesta; úsalo si cambias el PDF):

```bash
docker compose down -v
```

### Decisiones de empaquetado

- **Modelo de embeddings** → se descarga en *build time* y queda horneado en la
  imagen (`/app/.hf_cache`). Evita descargarlo en cada arranque y permite operar
  sin red. No es un volumen para que no lo "tape" un montaje vacío.
- **Índice de Chroma** → se genera como **paso inicial** (entrypoint idempotente)
  y se persiste en un **volumen nombrado**. Un índice horneado en build quedaría
  oculto por el volumen vacío del primer arranque, por eso se ingesta al arrancar.
- **torch CPU-only** → el RAG no usa GPU; se instala el wheel CPU de PyTorch para
  no arrastrar las librerías CUDA y reducir el tamaño de la imagen.

---

## Comandos de ejemplo — los tres casos de seguridad

Con el contenedor levantado, las consultas se lanzan con `docker compose exec`:

**(a) Un empleado pregunta por el salario del CEO → la Sección 3 no se recupera
(cae en `I do not know`):**

```bash
docker compose exec rag python src/query.py "What is the CEO annual salary?" --role employee
```

**(b) Alguien pide el SSN → bloqueado localmente, sin tocar el LLM ni la base
vectorial:**

```bash
docker compose exec rag python src/query.py "What is the CEO's SSN?"
```

**(c) Un admin consulta la compensación → recupera la Sección 3 con el SSN
redactado como `[REDACTED]`:**

```bash
docker compose exec rag python src/query.py "What is the executive compensation information?" --role admin
```

Consulta legítima de ejemplo (política de confidencialidad, Sección 1):

```bash
docker compose exec rag python src/query.py "policy on pasting confidential data into public AI models" --role employee
```

---

## Uso local sin Docker (opcional)

```bash
python -m venv venv
venv\Scripts\activate        # En Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python src/ingest.py                                   # Ingesta (una vez)
python src/query.py "..." --role employee              # Consulta
```

---

## Nota sobre el LLM

Por defecto se usa una respuesta simulada (MockLLM) que sintetiza a partir de los
fragmentos recuperados, **sin depender de ninguna API**. Para generar respuestas
en lenguaje natural respetando el grounding, se puede activar un modelo **local**
con Ollama editando `get_llm()` en `src/query.py` (instrucciones comentadas en el
propio código). Ollama corre en la máquina, así que ningún dato sale de la
infraestructura privada.
