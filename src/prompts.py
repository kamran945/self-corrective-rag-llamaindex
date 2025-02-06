from langchain_core.prompts import ChatPromptTemplate
from langchain import hub


grader_system_prompt = """You are a grader evaluating how relevant a retrieved document is to a user's query.
This does not require a strict assessment—your main goal is to filter out incorrect retrievals.
If the document includes keyword(s) or conveys a semantic meaning related to the query, mark it as relevant.
Assign a binary score of 'yes' or 'no' to indicate its relevance."""
grader_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", grader_system_prompt),
        ("human", "Retrieved document: \n\n {document} \n\n User query: {query}"),
    ]
)

rag_prompt = hub.pull("rlm/rag-prompt")
# rag_system_prompt = """
# You are given a **context** and a **question**.

# ### **Instructions:**
# - Answer the question **ONLY** using the information provided in the context.
# - **DO NOT** use any prior knowledge or assumptions.
# - If the context does not contain enough information, say:
#   **"The provided context does not have enough information to answer this question."**
# - **DO NOT** infer or add details that are not explicitly stated.

# """
# rag_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", rag_system_prompt),
#         (
#             "human",
#             "### **Context:**  {context} \n### **Question:** {question} \n### **Answer:**",
#         ),
#     ]
# )

hallucination_system_prompt = """You are a grader evaluating whether an LLM's response is strictly based on and supported by a given set of retrieved facts.
Assign a binary score of 'yes' or 'no.'
'No' means the response is fully grounded in the provided facts, with no additional information beyond reasonable rewording.
'Yes' means the response includes hallucinations, contradicts the facts, or introduces information not explicitly present in the retrieved facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_system_prompt),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {answer}"),
    ]
)

answer_system_prompt = """You are a grader evaluating whether an answer effectively addresses or resolves the query.
Assign a binary score of 'yes' or 'no.' 'Yes' indicates that the answer satisfactorily resolves the query."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", answer_system_prompt),
        ("human", "User query: \n\n {query} \n\n LLM generation: {answer}"),
    ]
)

transform_query_system_prompt = """You are a query rewriter that enhances an input query for optimal vectorstore retrieval.
Analyze the input to understand its underlying semantic intent and meaning before refining it."""
transform_query_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", transform_query_system_prompt),
        (
            "human",
            "Here is the initial query: \n\n {query} \n Formulate an improved query.",
        ),
    ]
)
