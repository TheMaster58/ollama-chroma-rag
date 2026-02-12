from langchain_text_splitters import RecursiveCharacterTextSplitter
from lab1_document_loader import load_document

def split_text(documents, chunk_size=1000, chunk_overlap=50):
    print(f"Splitting {len(documents)} documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    pdf_path = "A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf"
    
    # Load documents first
    docs = load_document(pdf_path)
    
    # Split documents
    chunks = split_text(docs)
    
    # Verify chunks
    if chunks:
        print("\n--- First Chunk ---")
        print(chunks[0].page_content)
        print("\n--- Metadata ---")
        print(chunks[0].metadata)
