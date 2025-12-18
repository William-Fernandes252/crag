from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from crag.graph.state import GraphState

_web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> GraphState:
    """Perform a web search based on the question in the graph state.

    Args:
        state (GraphState): The current state of the graph containing the question.

    Returns:
        GraphState: Updated graph state with documents from the web search.
    """
    question = state["question"]
    documents = state["documents"]

    results = _web_search_tool.invoke({"query": question})
    web_results_document = Document(
        page_content="\n\n".join(
            (result["content"] if isinstance(result, dict) else "")
            for result in results
        ),
        metadata={"source": "web_search"},
    )

    if documents is not None:
        documents.append(web_results_document)
    else:
        documents = [web_results_document]

    return {
        "documents": documents,
        "question": question,
        **state,
    }
