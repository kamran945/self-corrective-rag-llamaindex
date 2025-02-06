from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Event
from langchain_core.documents import Document
from typing import List


class RetrieveEvent(Event):
    query: str


class GradeDocsEvent(Event):
    query: str
    docs: List[NodeWithScore]


class TransformQueryEvent(Event):
    query: str


class GenerateResponseEvent(Event):
    query: str
    docs: str


class HallucinationCheckerEvent(Event):
    query: str
    docs: str
    answer: str


class GradeAnswerEvent(Event):
    query: str
    answer: str
