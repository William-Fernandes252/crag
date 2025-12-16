from langgraph.graph.state import END, StateGraph

from crag.graph.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from crag.graph.nodes import generate, grade_documents, retrieve, web_search
from crag.graph.state import GraphState


def _decide_to_generate(state: GraphState) -> str:
    """Decide whether to proceed to generation based on document grades."""
    if state["web_search"]:
        return WEB_SEARCH
    return GENERATE


builder = StateGraph(state_schema=GraphState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEB_SEARCH, web_search)
builder.add_node(GENERATE, generate)

builder.set_entry_point(RETRIEVE)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    _decide_to_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)

builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

graph = builder.compile()
