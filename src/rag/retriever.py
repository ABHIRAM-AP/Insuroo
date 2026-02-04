"""
RAG Retriever Module
Combines retrieval with LLM for question answering
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Dict


class InsuranceRAG:
    """Complete RAG system for insurance questions"""
    
    def __init__(
        self,
        vectorstore,
        api_key: str,
        model_name: str = "gemini-1.5-flash-latest",
        temperature: float = 0.3,
        max_tokens: int = 500
    ):
        """
        Initialize RAG system
        
        Args:
            vectorstore: Vector database to retrieve from
            api_key: Google API key
            model_name: Gemini model name
            temperature: LLM temperature (0=factual, 1=creative)
            max_tokens: Maximum response length
        """
        self.vectorstore = vectorstore
        
        print("🤖 Initializing RAG system...")
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Create prompt
        self._create_prompt()
        
        # Build chain
        self._build_chain()
        
        print("✅ RAG system ready!")
    
    def _create_prompt(self):
        """Create prompt template"""
        template = """You are a helpful insurance advisor for rural areas in India.
Explain insurance concepts in simple, clear language.

Use the following context to answer the question.
If you don't know from the context, say so - never make up information.

Context:
{context}

Question: {question}

Answer in simple language:"""
        
        self.PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def _build_chain(self):
        """Build RAG chain"""
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": self.PROMPT},
            return_source_documents=True
        )
    
    def query(self, question: str) -> Dict:
        """
        Ask a question
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\n{'='*60}")
        print(f"❓ {question}")
        print('='*60)
        
        result = self.qa_chain({"query": question})
        
        answer = result["result"]
        sources = result["source_documents"]
        
        print(f"\n💡 {answer}")
        print(f"\n📚 Used {len(sources)} sources")
        print('='*60)
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }


if __name__ == "__main__":
    # Test RAG
    from config.config import CHROMA_DIR, GOOGLE_API_KEY, MODEL_NAME
    from vector_store import VectorStoreManager
    
    # Load vectorstore
    vs_manager = VectorStoreManager(CHROMA_DIR)
    vectorstore = vs_manager.load_vectorstore()
    
    # Create RAG
    rag = InsuranceRAG(vectorstore, GOOGLE_API_KEY, MODEL_NAME)
    
    # Test
    rag.query("What is PMJJBY?")