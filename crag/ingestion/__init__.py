"""Ingestion module for PDF documents into a Qdrant vector store for RAG."""

import asyncio
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_core.document_loaders.base import BaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")


_COLLECTION_NAME = "crag"


_embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


_client = QdrantClient(url=_QDRANT_URL)
if not _client.collection_exists(_COLLECTION_NAME):
    _client.create_collection(
        collection_name=_COLLECTION_NAME,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )


_vector_store = QdrantVectorStore(
    collection_name=_COLLECTION_NAME,
    client=_client,
    embedding=_embeddings_model,
)


def _get_loader_for_path(path: Path) -> BaseLoader:
    """Get the appropriate document loader for the given path.

    Args:
        path (Path): The path to the document or directory.

    Returns:
        BaseLoader: The document loader for the given path.
    """
    if path.is_dir():
        return PyPDFDirectoryLoader(str(path))
    else:
        return PyPDFLoader(str(path))


async def ingest_pdfs(path: Path) -> None:
    """Ingest PDF documents from the given path into a vector store.

    Args:
        path (Path): The path to the document or directory of documents to ingest.

    Returns:
        VectorStore: The vector store containing the ingested documents.
    """
    loader = _get_loader_for_path(path)

    docs = await loader.aload()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunked_documents = text_splitter.split_documents(docs)

    await _vector_store.aadd_documents(chunked_documents)

    print(f"Ingested {len(chunked_documents)} chunks into the vector store.")


async def ingest_batch(paths: list[Path]) -> None:
    """Ingest PDF documents from multiple paths into a vector store.

    Args:
        paths (list[Path]): The list of paths to documents or directories to ingest.
    """
    async with asyncio.TaskGroup() as tg:
        for path in paths:
            tg.create_task(ingest_pdfs(path))
