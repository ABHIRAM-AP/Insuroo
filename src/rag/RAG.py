from rag.retriever import retrieve_documents

class RAGPipeline:

    def __init__(self, llm):
        self.llm = llm

    def ask(self, query):

        print("🔎 Retrieving documents...")

        docs = retrieve_documents(query)

        context = "\n".join(docs)

        prompt = f"""
        Use the following context to answer the question.

        Context:
        {context}

        Question:
        {query}
        """

        response = self.llm.generate(prompt)

        return response