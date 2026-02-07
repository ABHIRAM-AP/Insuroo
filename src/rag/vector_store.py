from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStoreManager:

  def __init__(self,persist_directory='/content/drive/MyDrive/Insuroo_AI/data/chroma_db'):
    self.persist_directory=persist_directory


    print(f"Loading Embedding models")
    self.embeddingModel = GoogleGenerativeAIEmbeddings(
    model="gemini-3-pro-preview"
)
  def create_vectorstore(self,chunks):
    vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddingModel,
            persist_directory=self.persist_directory,
            collection_name="insurance_docs"
        )
    vectorstore.persist()

    print(f"✅ Vector database created!")
    print(f"   Saved to: {self.persist_directory}")
    print(f"   Total embeddings: {len(chunks)}")
    print(f"   Size per embedding: 384 dimensions")
    print(f"   Total numbers stored: {len(chunks) * 384:,}")

    return vectorstore
  def load_vectorstore(self):
        """Load existing vector database"""
        print(f"Loading existing vector database...")

        vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddingModel,
            collection_name="insurance_docs"
        )

        print(f"✅ Vector database loaded from {self.persist_directory}")

        return vectorstore
  def test_search(self, vectorstore, query, k=3):
        """Test similarity search"""
        print(f"\n🔍 Testing search: '{query}'")
        print(f"   Looking for top {k} similar chunks...")

        results = vectorstore.similarity_search_with_score(query, k=k)

        print(f"\n📋 RESULTS:")
        for i, (doc, score) in enumerate(results, 1):
            similarity = 1 - score  # Convert distance to similarity
            print(f"\n  {i}. Similarity: {similarity:.2f}")
            print(f"     Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"     Page: {doc.metadata.get('page', 'N/A')}")
            print(f"     Content: {doc.page_content[:150]}...")

        return results