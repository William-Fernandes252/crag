from typing import Literal

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI
from pydantic import BaseModel, Field


class GradeAnswers(BaseModel):
    """Binary score for the correctness of a generated answer."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Answer is correct ('yes') or incorrect ('no').",
    )


_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0).with_structured_output(
    GradeAnswers
)

_system_prompt = """
You are an expert evaluator tasked with assessing whether an answer addresses/resolves a question. 

Give a 'yes' if the answer correctly and fully addresses the question, otherwise give a 'no' if it does not.
"""

_answer_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system_prompt),
        ("human", "User question: \n\n{question}\n\nGenerated Answer: {generation}"),
    ]
)

answer_grader = _answer_grade_prompt | _llm
