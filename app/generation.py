
# ==========================================
# CUSTOMER RAG - ANSWER GENERATION
# GROQ + GPT-OSS 120B
# ==========================================

import os
import time

from functools import lru_cache
from dotenv import load_dotenv

from langchain_groq import ChatGroq

from app.core.prompts import build_public_customer_prompt_value
from app.supabase_retrieval import retrieve_supabase_documents


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(BASE_DIR, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ==========================================
# STEP 1: CREATE GROQ MODEL
# ==========================================

@lru_cache(maxsize=1)
def create_llm():

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing from .env"
        )

    print("Initializing Groq model...")

    model = ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0,
        max_retries=2,
        timeout=60
    )

    return model


# ==========================================
# STEP 2: PREPARE CONTEXT
# ==========================================

def prepare_context(documents):

    context_parts = []
    sources = []

    for index, document in enumerate(
        documents,
        start=1
    ):

        if isinstance(document, dict):
            metadata = document.get("metadata") or {}
            source_name = (
                document.get("source")
                or metadata.get("source_file")
                or "Customer Knowledge Base"
            )
            page_number = document.get("page", metadata.get("page"))
            content = document.get("content", "")
        else:
            metadata = getattr(document, "metadata", {}) or {}
            source_name = metadata.get(
                "source_file",
                metadata.get("source", "Customer Knowledge Base")
            )
            page_number = metadata.get("page")
            content = getattr(document, "page_content", "")

        context_parts.append(
            f"""
SOURCE {index}

File: {source_name}

Page: {page_number}

Content:
{content}
"""
        )

        sources.append({
            "source": source_name,
            "page": page_number
        })

    return "\n".join(context_parts), sources


# ==========================================
# STEP 3: BUILD CUSTOMER PROMPT
# ==========================================

def build_prompt(question, context):

    return build_public_customer_prompt_value(
        question=question,
        context=context,
    )


# ==========================================
# STEP 4: EXTRACT RESPONSE TEXT
# ==========================================

def extract_answer(response):

    answer = response.content

    if isinstance(answer, list):

        text_parts = []

        for item in answer:

            if isinstance(item, dict):

                if item.get("type") == "text":

                    text_parts.append(
                        item.get("text", "")
                    )

            elif isinstance(item, str):

                text_parts.append(item)

        answer = "\n".join(text_parts)

    if not isinstance(answer, str):

        answer = str(answer)

    return answer.strip()


# ==========================================
# STEP 5: GENERATE CUSTOMER ANSWER
# ==========================================

def generate_answer(question: str):

    question = question.strip()

    if not question:

        raise ValueError(
            "Question cannot be empty."
        )

    # --------------------------------------
    # Retrieve customer knowledge
    # --------------------------------------

    retrieval_start = time.perf_counter()

    documents = retrieve_supabase_documents(
        question=question,
        role="CUSTOMER",
        top_k=3
    )

    retrieval_time = round(
        time.perf_counter() - retrieval_start,
        2
    )

    print(
        f"Customer retrieval completed in "
        f"{retrieval_time} seconds"
    )

    # --------------------------------------
    # Handle missing knowledge
    # --------------------------------------

    if not documents:

        return {
            "answer": (
                "I could not find relevant information in "
                "the Customer Knowledge Base. Please contact "
                "customer support for further assistance."
            ),
            "sources": [],
            "found_context": False,
            "retrieval_time": retrieval_time,
            "generation_time": 0
        }

    # --------------------------------------
    # Prepare context
    # --------------------------------------

    context, sources = prepare_context(documents)

    # --------------------------------------
    # Build prompt
    # --------------------------------------

    prompt = build_prompt(
        question,
        context
    )

    # --------------------------------------
    # Initialize Groq model
    # --------------------------------------

    model = create_llm()

    # --------------------------------------
    # Generate response
    # --------------------------------------

    print("Sending request to Groq...")

    generation_start = time.perf_counter()

    response = model.invoke(prompt)

    generation_time = round(
        time.perf_counter() - generation_start,
        2
    )

    print(
        f"Groq generation completed in "
        f"{generation_time} seconds"
    )

    # --------------------------------------
    # Extract answer
    # --------------------------------------

    answer = extract_answer(response)

    # --------------------------------------
    # Return final result
    # --------------------------------------

    return {
        "answer": answer,
        "sources": sources,
        "found_context": True,
        "retrieval_time": retrieval_time,
        "generation_time": generation_time
    }


# ==========================================
# STEP 6: TERMINAL CHAT
# ==========================================

if __name__ == "__main__":

    print("\n=================================")
    print("CUSTOMER RAG ASSISTANT STARTED")
    print("LLM: GROQ GPT-OSS 120B")
    print("=================================")

    print("Type 'exit' to stop.")

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":

            print("Assistant stopped.")

            break

        if not question:
            continue

        try:

            result = generate_answer(question)

            print("\nAssistant:")
            print(result["answer"])

            print("\nSources:")

            for source in result["sources"]:

                print(
                    source["source"],
                    "| Page:",
                    source["page"]
                )

            print(
                "\nRetrieval time:",
                result["retrieval_time"],
                "seconds"
            )

            print(
                "Generation time:",
                result["generation_time"],
                "seconds"
            )

        except Exception as error:

            print(
                "\nGeneration Error:",
                str(error)
            )
