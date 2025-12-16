from langchain_classic import hub
from langchain_core.output_parsers.string import StrOutputParser
from langchain_openai.chat_models.base import ChatOpenAI

_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0)
_prompt = hub.pull("rlm/rag-prompt")

generation_chain = _prompt | _llm | StrOutputParser()
