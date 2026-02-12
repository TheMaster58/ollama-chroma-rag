from langchain_community.vectorstores import Chroma
from lab1_document_loader import load_document
from lab2_text_splitter import split_text
from lab3_embedding import get_embedding_model
import shutil
import os

persist_directory = 'chroma_db'

def create_vector_db(chunks, embedding_model):
    if os.path.exists(persist_directory):
        print(f"Removing existing {persist_directory}...")
        shutil.rmtree(persist_directory)
        
    print("Creating Vector Database...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print(f"Vector Database created and persisted to {persist_directory}")
    return vectordb

if __name__ == "__main__":
    pdf_path = "A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf"
    
    # 1. Load
    docs = load_document(pdf_path)
    
    # 2. Split
    chunks = split_text(docs)
    
    # 3. Embed
    embedding_model = get_embedding_model()
    
    # 4. Store
    vectordb = create_vector_db(chunks, embedding_model)
    
    print(f"Collection count: {vectordb._collection.count()}")
