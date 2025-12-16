import os

from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from crag.graph.state import GraphState

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

embedding = OpenAIEmbeddings(model="text-embedding-3-small")

qdrant_client = QdrantClient(url=QDRANT_URL)
retriver = QdrantVectorStore(
    client=qdrant_client,
    embedding=embedding,
    collection_name="crag_collection",
).as_retriever()


def retrieve(state: GraphState) -> GraphState:
    """Retrieve relevant documents from Qdrant and update the graph state."""
    question = state["question"]
    if not question:
        raise ValueError("No query provided in the graph state input data.")

    documents = retriver.invoke(question)
    return {"documents": documents, "question": question, **state}
