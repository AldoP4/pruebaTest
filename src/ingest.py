"""
Ingesta del Employee Handbook para el sistema RAG.

Lee el PDF de `data/`, lo trocea en chunks, genera embeddings localmente
con un modelo de HuggingFace (sin API key) y los persiste en ChromaDB.

Uso:
    python src/ingest.py

Al terminar deja una colección de Chroma lista para consultarse en `chroma_db/`.
"""

import os
from pathlib import Path

# --- Rutas (relativas a la raíz del proyecto, no al cwd) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"
HF_CACHE_DIR = PROJECT_ROOT / ".hf_cache"

# La app de escritorio de Claude es una app empaquetada (MSIX): su AppData\Local
# está virtualizado a una ruta donde huggingface_hub no puede crear symlinks en
# Windows. Redirigimos la caché a una carpeta normal del proyecto y desactivamos
# los symlinks ANTES de importar las librerías de HuggingFace.
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re

import chromadb
import pypdf
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# --- Configuración ---
# Modelo de embeddings local (se descarga la primera vez, ~130 MB).
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "employee_handbook"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# --- CAPA 1 (ingesta): etiquetado de acceso por sección ---
# Un chunk se marca como "admin" si su sección lleva el propio marcador de
# acceso restringido del documento. NO hardcodeamos "Sección 3": leemos la
# etiqueta que el handbook ya trae, de modo que si mañana se restringe otra
# sección, el etiquetado la detecta sola.
ACCESS_ADMIN = "admin"
ACCESS_PUBLIC = "public"
ADMIN_MARKERS = ('visible_to": "admin', "RESTRICTED ACCESS")

# Separa el texto en secciones a partir de las cabeceras "SECTION N:".
SECTION_HEADER_RE = re.compile(r"SECTION\s+\d+\s*:")


def _collapse_ws(text: str) -> str:
    """Colapsa saltos de línea/espacios repetidos (ruido de pypdf) a un espacio."""
    return re.sub(r"\s+", " ", text).strip()


def _access_level(section_text: str) -> str:
    """Determina el nivel de acceso de una sección según sus marcadores."""
    if any(marker in section_text for marker in ADMIN_MARKERS):
        return ACCESS_ADMIN
    return ACCESS_PUBLIC


def _section_title(section_text: str) -> str:
    """Extrae un título legible: 'SECTION N: ...' hasta la 1ª subsección o '{'."""
    m = re.match(r"(SECTION\s+\d+\s*:\s*.+?)(?=\s+\d+\.\d|\s*\{)", section_text)
    return (m.group(1) if m else section_text[:60]).strip()


def _section_number(section_text: str) -> int:
    m = re.search(r"SECTION\s+(\d+)", section_text)
    return int(m.group(1)) if m else 0


def configure_settings() -> None:
    """Configura LlamaIndex para usar embeddings locales y sin LLM."""
    # `cache_folder` explícito: por defecto LlamaIndex usa
    # AppData\Local\llama_index\..., que MSIX virtualiza a una ruta donde falla
    # la escritura. Lo forzamos a una carpeta normal dentro del proyecto.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder=str(HF_CACHE_DIR),
    )
    # La ingesta solo necesita embeddings; desactivamos el LLM para no
    # requerir ninguna API key (LlamaIndex usa OpenAI por defecto).
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )


def load_documents():
    """Lee los PDFs de data/ y crea un Document por SECCIÓN, con metadata de acceso.

    Parseamos el PDF con pypdf y partimos el texto por cabeceras "SECTION N:".
    Cada sección se convierte en un Document etiquetado con `access_level`, lo
    que permite el filtrado por rol en la query (CAPA 1). Los chunks generados
    después heredan esta metadata.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta de datos: {DATA_DIR}")

    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise ValueError(f"No se encontraron PDFs en {DATA_DIR}")

    documents = []
    for pdf_path in pdf_paths:
        reader = pypdf.PdfReader(str(pdf_path))
        # Texto completo del PDF, con el whitespace ya limpio página a página.
        full_text = " ".join(_collapse_ws(p.extract_text()) for p in reader.pages)

        # Posiciones de cada cabecera "SECTION N:" para trocear por secciones.
        starts = [m.start() for m in SECTION_HEADER_RE.finditer(full_text)]

        # Texto anterior a la 1ª sección (portada/preámbulo) → público.
        if not starts or starts[0] > 0:
            preamble = full_text[: starts[0]] if starts else full_text
            if preamble.strip():
                documents.append(
                    Document(
                        text=preamble.strip(),
                        metadata={
                            "file_name": pdf_path.name,
                            "section_title": "Front Matter",
                            "section_number": 0,
                            "access_level": ACCESS_PUBLIC,
                        },
                    )
                )

        # Una sección por cada tramo entre cabeceras consecutivas.
        bounds = starts + [len(full_text)]
        for i in range(len(starts)):
            section_text = full_text[bounds[i] : bounds[i + 1]].strip()
            if not section_text:
                continue
            documents.append(
                Document(
                    text=section_text,
                    metadata={
                        "file_name": pdf_path.name,
                        "section_title": _section_title(section_text),
                        "section_number": _section_number(section_text),
                        "access_level": _access_level(section_text),
                    },
                )
            )

    # La metadata es para control de acceso y trazabilidad, NO para embeber:
    # la excluimos del texto que se convierte en embedding y del que ve el LLM.
    meta_keys = ["file_name", "section_title", "section_number", "access_level"]
    for doc in documents:
        doc.excluded_embed_metadata_keys = meta_keys
        doc.excluded_llm_metadata_keys = meta_keys

    admin_n = sum(1 for d in documents if d.metadata["access_level"] == ACCESS_ADMIN)
    print(
        f"Cargadas {len(documents)} secciones desde {DATA_DIR} "
        f"({admin_n} marcadas como '{ACCESS_ADMIN}', "
        f"{len(documents) - admin_n} '{ACCESS_PUBLIC}')"
    )
    return documents


def build_index(documents):
    """Trocea, embebe y persiste los documentos en ChromaDB."""
    # Cliente persistente de Chroma sobre disco.
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True,
    )

    # Distribución de niveles de acceso entre los chunks almacenados.
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    n_admin = sum(1 for m in metadatas if m.get("access_level") == "admin")
    print(
        f"Ingesta completada: {collection.count()} chunks almacenados "
        f"en la colección '{COLLECTION_NAME}' ({CHROMA_DIR})"
    )
    print(f"  Chunks con access_level=admin: {n_admin}")
    print(f"  Chunks con access_level=public: {len(metadatas) - n_admin}")
    return index


def main() -> None:
    configure_settings()
    documents = load_documents()
    build_index(documents)


if __name__ == "__main__":
    main()
