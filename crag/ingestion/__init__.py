import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


def ingest():
    """Ingest documents only if the collection is empty/missing."""

    collection_name = "crag_collection"
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 1. Initialize a persistent client (Local path or Server URL)
    # Using a local path ensures data survives after the script finishes.
    client = QdrantClient(url="http://localhost:6333")

    # 2. Check if collection exists and has documents
    collection_exists = client.collection_exists(collection_name)

    if collection_exists:
        # Check if it actually contains data (count > 0)
        doc_count = client.count(collection_name).count
        if doc_count > 0:
            print(
                f"Collection '{collection_name}' already contains {doc_count} documents. Skipping ingestion."
            )
            return QdrantVectorStore(
                client=client,
                collection_name=collection_name,
                embedding=embeddings,
            )

    # 3. If we reach here, we need to ingest
    print("Collection missing or empty. Starting ingestion...")

    all_docs = []
    for url in urls:
        loader = WebBaseLoader(url)
        docs = loader.load()
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(all_docs)

    # 4. Create the store using the SAME client
    vector_store = QdrantVectorStore.from_documents(
        split_docs,
        embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
    )

    print(f"Ingested {len(split_docs)} chunks into '{collection_name}'.")
    return vector_store
