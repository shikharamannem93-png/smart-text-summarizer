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
from langchain_community.document_loaders import TextLoader, UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader  # requires pypdf or pdfminer backend

from langchain_community.vectorstores import Chroma
import importlib
import subprocess
import sys
#!/usr/bin/env python3

from pathlib import Path
from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# Use the community huggingface embeddings wrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

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

# ingest_finance.py (relevant parts)


logger = logging.getLogger(__name__)

def chunk_and_index(documents: List[Document], persist_dir: str, collection_name: str = "finance"):
    """
    documents: list of LangChain Document objects (with .page_content and .metadata)
    persist_dir: where Chroma will persist its DB
    collection_name: optional collection name
    """

    # 1) create the embedding function using a local huggingface model
    #    'all-MiniLM-L6-v2' is small and fast for development.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 2) optionally split/prepare documents here (example uses documents already split)
    # If you need splitting, use your splitter:
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    # docs = []
    # for d in documents:
    #     chunks = text_splitter.split_text(d.page_content)
    #     for i, c in enumerate(chunks):
    #         docs.append(Document(page_content=c, metadata={**d.metadata, "chunk": i}))

    docs = documents  # or use the prepared/ split docs variable above

    # 3) create/Add to Chroma vectorstore using the HF embeddings
    try:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,            # pass huggingface embeddings object
            persist_directory=str(persist_dir),
            collection_name=collection_name
        )
        # Persist to disk (if using persistent chroma)
        try:
            vectordb.persist()
        except Exception:
            # some chroma versions call persist on the underlying client
            logger.debug("Chroma persist() call failed or not needed for this version.")
        logger.info(f"Indexed {len(docs)} documents into Chroma at {persist_dir}/{collection_name}")
        return vectordb
    except Exception as e:
        logger.exception("Failed to create vector DB with HuggingFace embeddings.")
        raise

def try_run_fetch_script(fetch_script: str, outdir: Path) -> bool:
    """
    Try to import and call a sensible function from the fetch script (preferred).
    If import fails, fallback to running the script as a subprocess.
    Returns True if any fetch attempt was performed, False otherwise.
    """
    mod_name = Path(fetch_script).stem

    # Try import & call (preferred, keeps it in-process)
    try:
        mod = importlib.import_module(mod_name)
        for fn in ("fetch_and_save", "fetch_and_write", "fetch_news", "fetch_articles", "main", "run"):
            if hasattr(mod, fn):
                func = getattr(mod, fn)
                try:
                    # prefer supplying output directory if the function accepts it
                    func(str(outdir))
                except TypeError:
                    func()
                return True
    except Exception:
        # Import failed or function raised; fall back to subprocess below
        pass

    # Fallback: run script as a subprocess and pass an --out/--outdir argument
    cmd = [sys.executable, fetch_script, "--out", str(outdir)]
    try:
        subprocess.run(cmd, check=True)
        return True
    except FileNotFoundError:
        print(f"Fetch script not found: {fetch_script}")
    except subprocess.CalledProcessError as e:
        print(f"Fetch script failed (exit {e.returncode}): {e}")
    except Exception as e:
        print(f"Unexpected error while running fetch script: {e}")

    return False


def main():
    parser = argparse.ArgumentParser(description="Ingest raw news into Chroma index.")
    parser.add_argument("--source", default="data/raw_news", help="Source folder containing documents to ingest")
    parser.add_argument("--collection", default="finance_docs", help="Chroma collection name")
    parser.add_argument("--fetch-script", default="fetch_news_finnhub.py",
                        help="Fetcher script to run/import (only used when --fetch is set)")
    parser.add_argument("--fetch", action="store_true", help="If set, run the fetch script before ingesting")
    args = parser.parse_args()

    outdir = Path(args.source)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.fetch:
        print(f"Running fetch script: {args.fetch_script} -> {outdir}")
        did_fetch = try_run_fetch_script(args.fetch_script, outdir)
        if not did_fetch:
            print("Warning: fetch attempt failed or was not performed. Continuing with whatever is in the source folder.")

    # Load documents from the folder (you already have this helper)
    docs = load_documents_from_folder(str(outdir))  # ensure this function is imported above
    if not docs:
        print("No documents found in", outdir)
        return

    # index into chroma (ensure chunk_and_index is imported above)

    chunk_and_index(
    docs,
    persist_dir="data/chroma_index",     # ✅ specify where to store your vector DB
    collection_name=args.collection
)
    print("Done.")


if __name__ == "__main__":
    main()
