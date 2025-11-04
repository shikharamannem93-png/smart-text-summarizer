# ingest_finance.py
# Simple ingestion to create a local Chroma collection of finance docs.
# Supports raw .txt, .html (basic), and .pdf (pdfminer).
# Usage: python ingest_finance.py --source data/raw_docs --collection finance_docs

import os
import argparse
from pathlib import Path
import json
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader  # requires pypdf or pdfminer backend

from langchain_community.vectorstores import Chroma

# Config
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIR = "chromadb_data"  # local directory (commit .gitignore this)

def load_documents_from_folder(source_folder: str):
    docs = []
    p = Path(source_folder)
    for f in sorted(p.rglob("*")):
        if f.is_dir():
            continue
        suf = f.suffix.lower()
        try:
            if suf in [".txt", ".md"]:
                loader = TextLoader(str(f), encoding="utf8")
            elif suf in [".html", ".htm"]:
                loader = UnstructuredHTMLLoader(str(f))
            elif suf in [".pdf"]:
                loader = PyPDFLoader(str(f))
            else:
                # skip unknown files
                continue
            loaded = loader.load()
            # Add metadata: filename and (optional) ticker/date from filename
            for d in loaded:
                d.metadata = d.metadata or {}
                d.metadata.update({"source_file": str(f.name)})
            docs.extend(loaded)
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    return docs

def chunk_and_index(docs, persist_directory=PERSIST_DIR, collection_name="finance_docs"):
    print(f"Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}) ...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunked_docs = []
    for d in docs:
        chunks = splitter.split_documents([d])
        chunked_docs.extend(chunks)

    print(f"Total chunks: {len(chunked_docs)}")
    # Initialize embeddings + chroma
    emb = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=emb,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    vectordb.persist()
    print("Indexing complete and persisted to", persist_directory)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder with raw documents")
    parser.add_argument("--collection", default="finance_docs", help="Chroma collection name")
    args = parser.parse_args()

    docs = load_documents_from_folder(args.source)
    if not docs:
        print("No documents found in", args.source)
        return
    chunk_and_index(docs, collection_name=args.collection)
    print("Done.")

if __name__ == "__main__":
    main()