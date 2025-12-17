from typing import Literal, cast

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel, Field

type Route = Literal["vector_store", "web_search"]


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vector_store", "web_search"] = Field(
        ...,
        description="Given a user question, choose, to route it to either 'vector_store' or 'web_search' based on the nature of the query.",
    )


_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0).with_structured_output(
    schema=RouteQuery
)

_system = """
You are an expert router that determines the most appropriate datasource to answer a user's query.

The vector store contains documents related to specific topics (like ai-agents, prompt engineering and adversarial attacks), while web search can provide up-to-date information on a wide range of subjects.
"""

_router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system),
        ("human", "User question: {question}"),
    ]
)

question_router = cast(RunnableSerializable[dict, RouteQuery], _router_prompt | _llm)
