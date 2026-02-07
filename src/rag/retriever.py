from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os

class InsuranceRAG:
    """RAG Retriever System"""

    def __init__(self, vectorstore, api_key=None):
        self.vectorstore = vectorstore

        if api_key is None:
            api_key = os.environ['GEMINI_API_KEY']

        # Initialize Gemini
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", 
            google_api_key=api_key,
            temperature=0.3
        )

        # Create prompt
        system_msg = """You are an insurance advisor for rural India.
Use ONLY the context to answer. Explain in simple language.

CONTEXT:
{context}"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{input}")
        ])

        # Build chain
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, doc_chain)

        print("✅ RAG system ready!")

    def query(self, question):
        print(f"\n❓ {question}")
        print("="*60)

        result = self.rag_chain.invoke({"input": question})
        answer = result.get("answer", "No answer")
        sources = result.get("context", [])

        print(f"\n💡 ANSWER:\n{answer}")
        print(f"\n📚 Sources: {len(sources)} chunks used")
        print("="*60)

        return result

