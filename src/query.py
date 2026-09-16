"""
Consulta del Employee Handbook (RAG) sobre el índice ya persistido en ChromaDB.

Flujo:
    1. Carga el índice de Chroma existente (NO re-ingesta).
    2. Recupera los top-k chunks más relevantes para la pregunta.
    3. Limpia el whitespace del texto recuperado y muestra de qué parte del
       documento (página) viene cada chunk.
    4. Sintetiza una respuesta con un LLM INTERCAMBIABLE. Por defecto usa una
       respuesta simulada extractiva (sin ninguna API key). Se puede cambiar a
       un modelo local tipo Ollama (ver `get_llm`).
    5. Grounding: si la respuesta no está en los chunks recuperados, responde
       exactamente "I do not know" en lugar de inventar.

Uso:
    python src/query.py "¿Cuál es la política de vacaciones?"
    python src/query.py "..." --top-k 5
"""

import os
from pathlib import Path

# --- Rutas (relativas a la raíz del proyecto, no al cwd) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"

# Misma redirección de caché que en la ingesta: la app empaquetada (MSIX)
# virtualiza AppData\Local a una ruta donde huggingface_hub falla en Windows.
# Debe fijarse ANTES de importar las librerías de HuggingFace.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import re

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- Configuración (debe coincidir con la de ingest.py) ---
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "employee_handbook"

# Cuántos chunks recuperar y umbral mínimo de similitud para considerarlos
# relevantes. El umbral es la primera línea de defensa del grounding: si ningún
# chunk supera este score, asumimos que la respuesta no está en el documento.
TOP_K = 3
# Umbral de similitud para el grounding. A 0.40 dejamos fuera coincidencias
# tangenciales de baja relevancia (p. ej. la portada del documento), de modo
# que si tras el filtrado por rol no queda contenido realmente relevante, la
# respuesta cae correctamente en "I do not know".
SIMILARITY_CUTOFF = 0.40

# System prompt de "grounding" exigido por el handbook: responder solo con el
# contexto recuperado y, si no está ahí, devolver exactamente "I do not know".
GROUNDING_SYSTEM_PROMPT = (
    "You are a strict assistant for the Global Tech Solutions Employee Handbook. "
    "Answer the question using ONLY the provided context excerpts. "
    "If the answer is not contained in the context, you MUST respond with exactly "
    '"I do not know" and nothing else. '
    "Never use outside knowledge and never invent information."
)

# Texto exacto que se devuelve cuando no hay contexto suficiente (grounding).
IDK = "I do not know"

# ======================================================================
# AIRLOCK / CAPA DE PRIVACIDAD — constantes
# ======================================================================
# Rol por defecto si no se especifica: el menos privilegiado.
DEFAULT_ROLE = "employee"
ADMIN_ROLE = "admin"

# CAPA 2 — bloqueo de entrada por PII. Si la pregunta pide SSN o tarjetas,
# se rechaza localmente. Mensaje exacto de rechazo:
BLOCKED_MSG = "This request has been blocked for security reasons."

# Palabras clave que indican una petición de PII sensible (ES/EN).
PII_KEYWORDS = (
    "ssn",
    "social security",
    "social-security",
    "credit card",
    "credit-card",
    "card number",
    "tarjeta de crédito",
    "tarjeta de credito",
    "número de tarjeta",
    "numero de tarjeta",
)

# CAPA 2 y 3 — patrones de PII.
#   SSN de EE. UU.: XXX-XX-XXXX
SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
#   Tarjeta de crédito: 13–16 dígitos, admitiendo espacios o guiones como
#   separadores (Visa/MC/Amex, etc.).
CC_REGEX = re.compile(r"\b(?:\d[ -]?){13,16}\b")

# Texto con el que se reemplaza cualquier PII detectada en la salida (CAPA 3).
REDACTION = "[REDACTED]"


def get_llm():
    """Devuelve el LLM a usar para sintetizar la respuesta. INTERCAMBIABLE.

    Por defecto devuelve None → la síntesis usa una respuesta simulada
    extractiva (`simulated_answer`), sin depender de ninguna API key ni de
    servicios externos, cumpliendo el requisito de inferencia local del
    handbook y su prohibición de enviar datos confidenciales a modelos
    públicos (p. ej. OpenAI).

    Para usar un modelo LOCAL con Ollama en el futuro, basta con:

        # 1) Instalar Ollama (https://ollama.com) y descargar un modelo:
        #      ollama pull llama3.1
        # 2) pip install llama-index-llms-ollama
        # 3) Descomentar:
        # from llama_index.llms.ollama import Ollama
        # return Ollama(model="llama3.1", request_timeout=120.0,
        #               system_prompt=GROUNDING_SYSTEM_PROMPT)
        #
        # Ollama corre 100% en local, así que ningún dato sale de la máquina.
    """
    return None


# ======================================================================
# CAPA 2 — Bloqueo de entrada por PII (pre-procesamiento)
# ======================================================================
def is_pii_request(question: str) -> bool:
    """Detecta si la PREGUNTA pide datos sensibles (SSN / tarjetas).

    IMPORTANTE: esta comprobación es puramente local (regex + keywords). Se
    ejecuta ANTES de tocar el retriever o el LLM, así que una petición de PII
    se rechaza sin consultar la base vectorial ni ningún modelo. Este es el
    requisito 3 de la prueba.
    """
    q = question.lower()
    if any(kw in q for kw in PII_KEYWORDS):
        return True
    # Si el propio texto de la pregunta ya contiene un SSN/tarjeta, también.
    return bool(SSN_REGEX.search(question) or CC_REGEX.search(question))


# ======================================================================
# CAPA 3 — Redacción de salida (Airlock, Sección 4.1)
# ======================================================================
def redact_pii(text: str) -> str:
    """Reemplaza cualquier SSN o número de tarjeta por [REDACTED].

    Última malla de seguridad: se aplica a TODO chunk recuperado antes de
    mostrarlo o de pasarlo al LLM, independientemente del rol. Así, aunque un
    chunk con PII llegue hasta aquí, el dato sensible nunca sale en claro.
    """
    text = SSN_REGEX.sub(REDACTION, text)
    text = CC_REGEX.sub(REDACTION, text)
    return text


def clean_and_redact(text: str) -> str:
    """Pipeline de texto recuperado: limpia whitespace y luego redacta PII."""
    return redact_pii(clean_whitespace(text))


def clean_whitespace(text: str) -> str:
    """Colapsa el whitespace del texto recuperado.

    pypdf, con PDFs generados por Google Docs, separa a menudo cada palabra
    con secuencias como "\\n \\n". Reducimos cualquier bloque de espacios,
    tabs y saltos de línea a un único espacio y recortamos los extremos, para
    que el texto recuperado sea legible y se sintetice mejor.
    """
    return re.sub(r"\s+", " ", text).strip()


def load_index() -> VectorStoreIndex:
    """Carga el índice desde la colección de Chroma ya persistida (sin re-ingesta)."""
    # Mismo modelo de embeddings que en la ingesta (con cache_folder explícito).
    # `query_instruction`: los modelos BGE recuperan mucho mejor si la CONSULTA
    # lleva este prefijo, mientras que los pasajes (ya embebidos en la ingesta)
    # NO lo llevan. Es el uso asimétrico recomendado por el modelo.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder=str(HF_CACHE_DIR),
        query_instruction="Represent this sentence for searching relevant passages:",
    )
    # MockLLM como LLM por defecto de LlamaIndex: evita que cualquier operación
    # interna intente llamar a OpenAI (que es el LLM por defecto de la librería).
    Settings.llm = get_llm() or MockLLM()

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No existe {CHROMA_DIR}. Ejecuta primero la ingesta: python src/ingest.py"
        )

    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # from_vector_store reconstruye el índice consultable a partir de los
    # vectores ya almacenados, sin volver a leer ni embeber el PDF.
    return VectorStoreIndex.from_vector_store(vector_store)


# ======================================================================
# CAPA 1 — Filtrado por rol (metadata filtering)
# ======================================================================
def build_role_filters(user_role: str):
    """Filtro de metadata según el rol del usuario.

    - admin: sin filtro, ve todos los chunks.
    - cualquier otro rol (p. ej. 'employee'): se EXCLUYEN del retrieval los
      chunks con access_level == 'admin'. La exclusión ocurre en la propia
      consulta a Chroma, así que esos chunks nunca llegan al pipeline: un
      empleado que pregunte por el salario del CEO no recupera nada de la
      Sección 3 y termina cayendo en el "I do not know".
    """
    if user_role == ADMIN_ROLE:
        return None
    return MetadataFilters(
        filters=[
            MetadataFilter(
                key="access_level", value="admin", operator=FilterOperator.NE
            )
        ]
    )


def retrieve(index: VectorStoreIndex, question: str, user_role: str, top_k: int = TOP_K):
    """Recupera top-k chunks aplicando filtro por rol (CAPA 1) y umbral de similitud."""
    retriever = index.as_retriever(
        similarity_top_k=top_k, filters=build_role_filters(user_role)
    )
    nodes = retriever.retrieve(question)
    # Nos quedamos solo con los chunks suficientemente similares (grounding).
    return [n for n in nodes if (n.score or 0.0) >= SIMILARITY_CUTOFF]


def format_source(node: NodeWithScore) -> str:
    """Describe de qué parte del documento proviene un chunk."""
    meta = node.metadata or {}
    seccion = meta.get("section_title") or f"Sección {meta.get('section_number', '?')}"
    fname = meta.get("file_name") or meta.get("file_path") or "documento"
    nivel = meta.get("access_level", "?")
    return f"{fname} — {seccion} [access={nivel}] (score={node.score:.3f})"


def simulated_answer(question: str, nodes) -> str:
    """Respuesta SIMULADA (sin LLM real): síntesis extractiva de los chunks.

    Si no hay chunks relevantes, aplica el grounding y devuelve "I do not know".
    En caso contrario, une los fragmentos recuperados (ya limpios) como una
    respuesta basada exclusivamente en el contexto. No genera lenguaje nuevo:
    solo reorganiza lo recuperado, que es justo lo que se espera de un mock.
    """
    if not nodes:
        return IDK

    fragmentos = []
    for i, n in enumerate(nodes, start=1):
        # CAPA 3: el texto se limpia y se redacta la PII antes de mostrarse.
        texto = clean_and_redact(n.get_content())
        fragmentos.append(f"[{i}] {texto}")

    return (
        "(Respuesta simulada — síntesis extractiva a partir del contexto "
        "recuperado; sin LLM real)\n\n" + "\n\n".join(fragmentos)
    )


def llm_answer(llm, question: str, nodes) -> str:
    """Respuesta con un LLM real (p. ej. Ollama), respetando el grounding.

    Solo se usa si `get_llm()` devuelve un LLM. Construye el contexto con los
    chunks limpios y le pide al modelo que responda ciñéndose a él.
    """
    if not nodes:
        return IDK

    # CAPA 3: también se redacta la PII ANTES de construir el contexto del LLM,
    # para que ningún dato sensible llegue siquiera al modelo.
    contexto = "\n\n".join(
        f"[{i}] {clean_and_redact(n.get_content())}" for i, n in enumerate(nodes, start=1)
    )
    prompt = (
        f"{GROUNDING_SYSTEM_PROMPT}\n\n"
        f"Context:\n{contexto}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
    return str(llm.complete(prompt)).strip()


def answer_question(
    index: VectorStoreIndex,
    question: str,
    user_role: str = DEFAULT_ROLE,
    top_k: int = TOP_K,
):
    """Orquesta las 3 capas de la Airlock + síntesis. Devuelve (respuesta, nodos).

    Orden de las capas (importa):
      CAPA 2 (bloqueo de entrada)  ->  CAPA 1 (filtro por rol)  ->  síntesis
      con CAPA 3 (redacción de salida) incrustada en la síntesis.
    """
    # --- CAPA 2: bloqueo de entrada por PII, ANTES de tocar retrieval o LLM. ---
    if is_pii_request(question):
        # No se llama ni a Chroma ni al LLM: rechazo puramente local.
        return BLOCKED_MSG, []

    # --- CAPA 1: recuperación filtrada por rol. ---
    nodes = retrieve(index, question, user_role, top_k=top_k)

    # --- Síntesis (con CAPA 3 de redacción aplicada dentro). ---
    llm = get_llm()
    if llm is None:
        respuesta = simulated_answer(question, nodes)
    else:
        respuesta = llm_answer(llm, question, nodes)
    return respuesta, nodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Consulta el Employee Handbook (RAG).")
    parser.add_argument("question", help="Pregunta en lenguaje natural.")
    parser.add_argument(
        "--role",
        default=DEFAULT_ROLE,
        help=f"Rol del usuario para el control de acceso (def. '{DEFAULT_ROLE}'). "
        f"Usa '{ADMIN_ROLE}' para ver las secciones restringidas.",
    )
    parser.add_argument(
        "--top-k", type=int, default=TOP_K, help=f"Nº de chunks a recuperar (def. {TOP_K})."
    )
    args = parser.parse_args()

    index = load_index()
    respuesta, nodes = answer_question(
        index, args.question, user_role=args.role, top_k=args.top_k
    )

    print(f"\nPregunta: {args.question}  (rol: {args.role})\n")
    print("=== Respuesta ===")
    print(respuesta)

    print("\n=== Fuentes (chunks recuperados) ===")
    if respuesta == BLOCKED_MSG:
        print("(petición bloqueada en la CAPA 2: no se consultó Chroma ni el LLM)")
    elif not nodes:
        print("(ninguna: ningún chunk superó el umbral de similitud)")
    else:
        for i, n in enumerate(nodes, start=1):
            print(f"[{i}] {format_source(n)}")


if __name__ == "__main__":
    main()
