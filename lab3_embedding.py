import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError:
    OllamaEmbeddings = None

# Load environment variables
load_dotenv()

def get_embedding_model():
    if OllamaEmbeddings:
        print("Using OllamaEmbeddings (nomic-embed-text)")
        # Make sure you have pulled this model: ollama pull nomic-embed-text
        return OllamaEmbeddings(model="nomic-embed-text")
    else:
        print("Using HuggingFaceEmbeddings (all-mpnet-base-v2) as fallback")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if __name__ == "__main__":
    embedding_model = get_embedding_model()
    
    text = "This is a test sentence to check embeddings."
    embedding = embedding_model.embed_query(text)
    
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 dimensions: {embedding[:5]}")
