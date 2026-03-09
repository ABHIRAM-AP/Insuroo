from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
import os

load_dotenv()

# class InsuranceRAG:
#     """RAG Retriever System"""

#     def __init__(self, vectorstore, api_key=None):
#         self.vectorstore = vectorstore

#         if api_key is None:
#             api_key = os.environ['GEMINI_API_KEY']

#         # Initialize Gemini
#         self.llm = ChatGoogleGenerativeAI(
#             model="gemini-2.5-flash", 
#             google_api_key=api_key,
#             temperature=0.3
#         )

#         # Create prompt
#         system_msg = """You are an insurance advisor for rural India.
# Use ONLY the context to answer. Explain in simple language.

# CONTEXT:
# {context}"""

#         self.prompt = ChatPromptTemplate.from_messages([
#             ("system", system_msg),
#             ("human", "{input}")
#         ])

#         # Build chain
#         self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
#         doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
#         self.rag_chain = create_retrieval_chain(self.retriever, doc_chain)

#         print("✅ RAG system ready!")

#     def query(self, question):
#         result = self.rag_chain.invoke({"input": question})
        
#         sources = result.get("context", [])

#         return {
#             "answer":result.get("answer", "No answer"),
#             "sources":[
#                 {
#                     "source": doc.metadata.get("source", "unknown"),
#                     "page": doc.metadata.get("page", "N/A")
#                 }
#                 for doc in result.get("context", [])
#             ]
#         }
class InsuranceRAG:

    def __init__(self, vectorstore, api_key=None):

        self.vectorstore = vectorstore

        if api_key is None:
            api_key = os.environ["GEMINI_API_KEY"]

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        system_msg = """
You are an insurance advisor for rural India.

Use ONLY the provided context.

Rules:
- Explain in very simple language
- Maximum 120 words
- Friendly assistant tone
- If answer not found say you don't know

CONTEXT:
{context}
"""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", "{input}")
        ])

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        doc_chain = create_stuff_documents_chain(self.llm, self.prompt)

        self.rag_chain = create_retrieval_chain(self.retriever, doc_chain)

        print("✅ RAG system ready!")

    def query(self, question):

        result = self.rag_chain.invoke({"input": question})

        sources = result.get("context", [])

        return {
            "answer": result.get("answer", "No answer"),
            "sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "N/A")
                }
                for doc in sources
            ]
        }