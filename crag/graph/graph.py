from typing import Literal

from langgraph.graph.state import END, StateGraph

from crag.graph.constants import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEB_SEARCH
from crag.graph.nodes import generate, grade_documents, retrieve, web_search
from crag.graph.nodes.chains.answer_grader import answer_grader
from crag.graph.nodes.chains.hallucination_grader import hallucination_grader
from crag.graph.nodes.chains.router import question_router
from crag.graph.state import GraphState


def _decide_to_generate(state: GraphState) -> str:
    """Decide whether to proceed to generation based on document grades."""
    if state["web_search"]:
        return WEB_SEARCH
    return GENERATE


def _route_question(state: GraphState) -> str:
    """Route a question to the appropriate datasource.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        Route: The chosen route for the question.
    """
    question = state["question"]
    source = question_router.invoke({"question": question})
    if source.datasource == "vector_store":
        return RETRIEVE
    elif source.datasource == "web_search":
        return WEB_SEARCH
    return RETRIEVE


def _grade_grounded_in_documents_and_question(
    state: GraphState,
) -> Literal["useful", "not_useful", "not_supported"]:
    """Verify if the answer is grounded in the provided documents and question.

    Args:
        state: The current state of the graph.

    Returns:
        A string indicating if result:
            - "useful" if the answer is grounded and correct.
            - "not_useful" if the answer is grounded but incorrect.
            - "not_supported" if the answer is not grounded in the documents.
    """

    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    hallucination_score = hallucination_grader.invoke(
        {
            "documents": documents,
            "generation": generation,
        }
    )

    if hallucination_score.binary_score == "yes":
        answer_score = answer_grader.invoke(
            {
                "question": question,
                "generation": generation,
            }
        )
        if answer_score.binary_score == "yes":
            return "useful"
        else:
            return "not_useful"
    else:
        return "not_supported"


builder = StateGraph(state_schema=GraphState)

builder.add_node(RETRIEVE, retrieve)
builder.add_node(GRADE_DOCUMENTS, grade_documents)
builder.add_node(WEB_SEARCH, web_search)
builder.add_node(GENERATE, generate)

builder.set_conditional_entry_point(
    _route_question,
    path_map={
        RETRIEVE: RETRIEVE,
        WEB_SEARCH: WEB_SEARCH,
    },
)
builder.add_edge(RETRIEVE, GRADE_DOCUMENTS)
builder.add_conditional_edges(
    GRADE_DOCUMENTS,
    _decide_to_generate,
    path_map={
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)
builder.add_conditional_edges(
    GENERATE,
    _grade_grounded_in_documents_and_question,
    path_map={
        "useful": END,
        "not_useful": WEB_SEARCH,
        "not_supported": GENERATE,
    },
)

builder.add_edge(WEB_SEARCH, GENERATE)
builder.add_edge(GENERATE, END)

graph = builder.compile()
