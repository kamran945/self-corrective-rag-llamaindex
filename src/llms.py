from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)


from src.prompts import (
    grader_prompt,
    rag_prompt,
    hallucination_prompt,
    answer_prompt,
    transform_query_prompt,
)
from src.output_schema import GradeDocuments, GradeHallucinations, GradeAnswer


llm = ChatGroq(model="llama3-70b-8192", temperature=0)
retrieval_grader = grader_prompt | llm.with_structured_output(GradeDocuments)

rag_chain = rag_prompt | llm | StrOutputParser()

hallucination_grader = hallucination_prompt | llm.with_structured_output(
    GradeHallucinations
)

answer_grader = answer_prompt | llm.with_structured_output(GradeAnswer)

query_transformer = transform_query_prompt | llm | StrOutputParser()
