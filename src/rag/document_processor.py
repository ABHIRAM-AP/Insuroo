"""
Loads PDF's and split them into chunks (1st Step)
"""

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List
import os 

class DocumentProcessor:

    def __init__(self, docs_path:str,chunk_size:int=1000,chunk_overlap:int=200):
        self.docs_path = Path(docs_path)
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap

        if not self.docs_path.exists():
            raise FileNotFoundError(f"Documents path not found: {docs_path}")

    def load_documents(self):
        print(f"Loading files from: {self.docs_path}")

      

        loader = DirectoryLoader(
            path=str(self.docs_path),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )

        documents = loader.load()

        if len(documents) == 0:
            raise ValueError(f"No PDF files found in {self.docs_path}")


        return documents

    """ For Splitting the documents into smaller chunks with overlap """
    def split_into_chunks(self, documents):
        print(f"Splitting {len(documents)} pages into chunks .....\n")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks")
        return chunks

    def process(self):
        print("=" * 60)
        print("DOCUMENT PROCESSING PIPELINE")
        print("=" * 60)

        documents = self.load_documents()
        chunks = self.split_into_chunks(documents)

        print("=" * 60)
        print(f"✅ PIPELINE COMPLETE: {len(chunks)} chunks ready")
        print("=" * 60)

        return chunks


if __name__ == "__main__":

    from config.config import DOCS_DIR
    processor = DocumentProcessor(docs_path=DOCS_DIR)
    chunks = processor.process()

    print("\n Sample Chunk:")
    print(f"Source: {chunks[0].metadata.get('source')}")
    print(f"Content: {chunks[0].page_content[:200]}.......")

"""
    For testing load documents function
        # for i, doc in enumerate(documents):
        #     print(f"\n📄 Document {i + 1}")
        #     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        #     print(f"Content preview: {doc.page_content[:100]}")
        #     print(f"Content length: {len(doc.page_content)}")

"""