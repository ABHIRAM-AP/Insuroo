from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import os
from typing import List, Dict


class InsuraceRAG:
    def __init__(
        self,
        vectorstore,
        api_key: str = None,
        model_name: str = "gemini-1.5-flash-latest",
        temperature: float = 0.3,
        max_token: int = 500,
        num_chunks: int = 3,
    ):
        print("\n" + "=" * 60)
        print("🤖 INITIALIZING RAG SYSTEM")
        print("=" * 60)
        self.vectorstore = vectorstore
        self.num_chunks = num_chunks

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_tokens=max_token,
        )
        print("LLM INITIALIZED\n")

        self._create_prompt_template()
        self._build_chain()
        print("\n" + "=" * 60)
        print("✅ RAG SYSTEM READY!")
        print("=" * 60)

def _create_prompt_template():
    template = """You are a helpful and knowledgeable insurance advisor for rural India.
Your goal is to help people understand insurance schemes in simple, clear language.

IMPORTANT INSTRUCTIONS:
1. Use ONLY the context provided below to answer
2. If the context doesn't contain enough information, say "I don't have enough information about that in the documents"
3. Explain in simple language - avoid jargon
4. If you use technical terms, explain them in simple words
5. Be specific - mention scheme names, amounts, eligibility clearly
6. Keep answers concise but complete

CONTEXT FROM INSURANCE DOCUMENTS:
{context}

QUESTION: {question}

HELPFUL ANSWER IN SIMPLE LANGUAGE:"""
        
