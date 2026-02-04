import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# vector store
from langchain_community.vectorstores import Chroma

# embeddings (using huggingface)
from langchain.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_PATH = "./data"
DB_PATH = "./chroma_db"


def create_vector_db():
    print("Initializing phase 2: DATA ingestion...")


    # check if data folder exists
    if not os.path.exists(DATA_PATH):
        print(f" Error: '{DATA_PATH}' folder does not found!")
        return

    print(' Loading pdfs from data folder...')
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()

    if not documents:
        print("Documents not found please upload or recheck it!....")
        return

    print(f" loaded {len(documents)} pages.")

    # creating chunks
    print(" Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)
    print(f"created {len(chunks)} text chunks. ")

    # 3. creating embeddings and save
    print("🧠 Creating Embeddings using HuggingFace (Local)...")

    # huggingface model implementation
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Saving to ChromaDB... wait for a while it will take some time...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_PATH
    )

    print(f" Success! Database ready at '{DB_PATH}'")

if __name__ == "__main__":
    create_vector_db()
