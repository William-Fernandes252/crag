from typing import Literal

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Relevance score indicating if the documents are relevant ('yes') or not ('no').",
    )


_llm = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(
    GradeDocuments
)

_system_prompt = """
You are an expert evaluator tasked with assessing the relevance of retrieved documents to a given question.

If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.

Respond with 'yes' if the documents are relevant to the question, or 'no' if they are not.
"""

_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system_prompt),
        ("human", "Retrieved document: \n\n{document}\n\nQuestion: {question}"),
    ]
)

retrieval_grader = _grade_prompt | _llm
