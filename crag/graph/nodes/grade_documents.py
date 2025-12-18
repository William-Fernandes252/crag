"""Acts as the Retrieval Evaluator described in the CRAG paper.

It iterates through retrieved documents and uses a LLM-based chain to score their relevance.
"""

from typing import cast

from crag.graph.nodes.chains.retrieval_grader import GradeDocuments, retrieval_grader
from crag.graph.state import GraphState


def grade_documents(state: GraphState) -> GraphState:
    """Determine whether the retrieved documents are relevant to the question.

    If any document is not relevant, it sets the flag to run a web search.

    Args:
        state (GraphState): The current state of the graph containing the question and documents.

    Returns:
        GraphState: Updated graph state with the web_search flag set based on document relevance.
    """
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for doc in documents:
        result = cast(
            GradeDocuments,
            retrieval_grader.invoke(
                {"document": doc.page_content, "question": question}
            ),
        )
        if result.binary_score == "yes":
            filtered_docs.append(doc)
        else:
            web_search = True

    return {**state, "documents": filtered_docs, "web_search": web_search}
