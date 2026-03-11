import os
import shutil
import stat
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DATA_PATH = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

load_dotenv(dotenv_path=ENV_FILE)


def _remove_readonly(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def extract_metadata_from_filename(filename: str):
    lowered = filename.lower()
    doc_type = "Manual"
    year = "2019"

    if "special provisions" in lowered or "om" in lowered or "amendment" in lowered or "preference" in lowered:
        doc_type = "Office Memorandum / Amendment"
        year = "2025"
    elif "gfr" in lowered:
        doc_type = "GFR Rule"
        year = "2017"

    return doc_type, year


def create_vector_db():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_PATH}")

    pdf_files = list(DATA_PATH.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {DATA_PATH}")

    loader = PyPDFDirectoryLoader(str(DATA_PATH))
    documents = loader.load()

    for document in documents:
        source_file = os.path.basename(document.metadata.get("source", ""))
        doc_type, year = extract_metadata_from_filename(source_file)
        document.metadata["doc_type"] = doc_type
        document.metadata["year"] = year

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH, onerror=_remove_readonly)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_PATH),
    )

    return str(DB_PATH)
