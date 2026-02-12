from langchain_community.document_loaders import PyPDFLoader

def load_document(file_path):
    print(f"Loading document from: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")
    return documents

if __name__ == "__main__":
    # PDF file path
    pdf_path = "A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf"
    
    docs = load_document(pdf_path)
    
    # Print content of the first page to verify
    if docs:
        print("\n--- Content of the first page ---")
        print(docs[0].page_content[:500] + "...")
        print("\n--- Metadata of the first page ---")
        print(docs[0].metadata)
