from typing import Literal, cast

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class GradeHallucinations(BaseModel):
    """Binary score for hallucinations present in a generated answer."""

    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Answer is grounded in the facts ('yes') or not ('no').",
    )


_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0).with_structured_output(
    GradeHallucinations
)

_system_prompt = """
You are a grader assesing whether an LLM generation is grounded/supported by a set of retrieved facts and information.

Give a 'yes' if the answer is fully supported by the facts, otherwise give a 'no' if there are any hallucinations or unsupported claims in the answer.
"""


_hallucination_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _system_prompt),
        ("human", "Facts: \n\n{documents}\n\LLM generation: {generation}"),
    ]
)


hallucination_grader = cast(
    RunnableSerializable[dict, GradeHallucinations], _hallucination_grade_prompt | _llm
)
