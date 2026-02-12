from langchain_community.vectorstores import Chroma
from lab3_embedding import get_embedding_model
import os

persist_directory = 'chroma_db'

def get_retriever(k=3):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector Database not found at {persist_directory}. Please run lab4_vector_db.py first.")
        
    embedding_model = get_embedding_model()
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return retriever

if __name__ == "__main__":
    try:
        retriever = get_retriever(k=2)
        
        query = "What is Low-Rank Adaptation?"
        print(f"Query: {query}")
        
        docs = retriever.invoke(query)
        
        for i, doc in enumerate(docs):
            print(f"\n--- Result {i+1} ---")
            print(doc.page_content[:400] + "...")
            print(f"Source: {doc.metadata}")
            
    except Exception as e:
        print(f"Error: {e}")
