import os
from typing import List
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    Document,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    get_response_synthesizer,
)
from llama_index.core import DocumentSummaryIndex
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.agent.react import ReActAgent
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.core.postprocessor.rankGPT_rerank import RankGPTRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.utils.workflow import draw_all_possible_flows


from dotenv import load_dotenv, find_dotenv

# Load the API keys from .env
load_dotenv(find_dotenv(), override=True)


from src.event_schema import (
    RetrieveEvent,
    GradeDocsEvent,
    TransformQueryEvent,
    GenerateResponseEvent,
    HallucinationCheckerEvent,
    GradeAnswerEvent,
)
from src.llms import (
    retrieval_grader,
    rag_chain,
    hallucination_grader,
    answer_grader,
    query_transformer,
)


class SelfCorrectiveRAG(Workflow):

    Settings.chunk_size = 250
    Settings.chunk_overlap = 0
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Groq(model="deepseek-r1-distill-llama-70b")

    # Load documents from the specified directory

    def load_or_create_index(self, directory_path, persist_dir):
        # Check if the index already exists
        if os.path.exists(persist_dir):
            print("Loading existing index...")
            # Load the index from disk
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            index = load_index_from_storage(storage_context)
        else:
            print("Creating new index...")
            documents = SimpleDirectoryReader(directory_path).load_data()

            # Create a new index from the documents
            index = VectorStoreIndex.from_documents(documents)

            # Persist the index to disk
            index.storage_context.persist(persist_dir=persist_dir)

        return index

    @step(pass_context=True)
    async def retrieve_docs(
        self, ctx: Context, ev: StartEvent | RetrieveEvent
    ) -> GradeDocsEvent:
        print("---- retrieve_docs ----")

        ctx.data["index"] = self.load_or_create_index(
            "E:\python projects\self-corrective-rag-llamaindex\data\posts",
            # "E:\python projects\self-corrective-rag-llamaindex\data\papers",
            "E:\python projects\self-corrective-rag-llamaindex\data\storage",
        )
        index = ctx.data["index"]

        retriever = index.as_retriever(similarity_top_k=2)
        docs = await retriever.aretrieve(ev.query)

        print("docs:", docs)
        print()

        return GradeDocsEvent(query=ev.query, docs=docs)

    @step(pass_context=True)
    async def grade_docs(
        self, ctx: Context, ev: GradeDocsEvent
    ) -> TransformQueryEvent | GenerateResponseEvent:
        print("---- grade_docs ----")

        # Score each doc
        filtered_docs = []
        for d in ev.docs:
            score = retrieval_grader.invoke(
                {"query": ev.query, "document": d.get_text()}
            )
            grade = score.binary_score
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
        if filtered_docs:

            def format_docs(docs: List[NodeWithScore]):
                return "\n\n".join(doc.get_text() for doc in docs)

            print(f"doc string: {format_docs(filtered_docs)}")

            return GenerateResponseEvent(
                query=ev.query, docs=format_docs(filtered_docs)
            )
        else:
            return TransformQueryEvent(query=ev.query)

    @step(pass_context=True)
    async def transform_query(
        self, ctx: Context, ev: TransformQueryEvent
    ) -> RetrieveEvent:
        print("---- transform_query ----")
        transformed_query = query_transformer.invoke({"query": ev.query})
        return RetrieveEvent(query=transformed_query)

    @step(pass_context=True)
    async def generate_response(
        self, ctx: Context, ev: GenerateResponseEvent
    ) -> HallucinationCheckerEvent:
        print("---- generate_response ----")
        from src.prompts import rag_prompt

        print(f"RAG PROMPT: {rag_prompt}")
        answer = rag_chain.invoke({"context": ev.docs, "question": ev.query})
        print("Generated response:", answer)

        return HallucinationCheckerEvent(query=ev.query, docs=ev.docs, answer=answer)

    @step(pass_context=True)
    async def hallucination_checker(
        self, ctx: Context, ev: HallucinationCheckerEvent
    ) -> GradeAnswerEvent | TransformQueryEvent:
        print("---- hallucination_checker ----")
        from src.prompts import hallucination_prompt

        print(f"PROMPT: {hallucination_prompt}")
        print(f"documents: {ev.docs}, answer: {ev.answer}")
        response = hallucination_grader.invoke(
            {"documents": ev.docs, "answer": ev.answer}
        )
        print("Hallucination Checker response:", response)

        if response.binary_score == "no":
            return GradeAnswerEvent(query=ev.query, answer=ev.answer)
        return TransformQueryEvent(query=ev.query)

    @step(pass_context=True)
    async def grade_answer(
        self, ctx: Context, ev: GradeAnswerEvent
    ) -> TransformQueryEvent | StopEvent:
        print("---- grade_answer ----")
        response = answer_grader.invoke({"query": ev.query, "answer": ev.answer})
        print("Answer Grader response:", response)
        if response.binary_score == "yes":
            return StopEvent(result="END")

        return TransformQueryEvent(query=ev.query)
