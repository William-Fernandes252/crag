from typing import TypedDict

from langchain_core.documents.base import Document


class GraphState(TypedDict):
    """ "Represents the state of the graph.

    Attributes:
        question (str): The current question being processed.
        generation (str): The current generation or output related to the question.
        web_search (bool): Whether to perform a web search for additional information.
        documents (list[Document]): A list of relevant documents retrieved for the question.
    """

    question: str
    generation: str
    web_search: bool
    documents: list[Document]
