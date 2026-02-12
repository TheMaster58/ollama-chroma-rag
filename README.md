# RAG with Gradio and Ollama for Research Papers

This project demonstrates a complete Retrieval Augmented Generation (RAG) pipeline for questioning research papers (specifically the LoRA paper).

It is built using:
- **LangChain**: For document loading, splitting, and prompt management.
- **ChromaDB**: As the vector store for efficient similarity search.
- **Ollama**: For local LLM inference (Llama 3) and embeddings (nomic-embed-text).
- **Gradio**: For the chat interface.

## Prerequisites

1.  **Python 3.10+** (Recommend 3.11)
2.  **Ollama**: Installed and running locally.
    - Download from [ollama.com](https://ollama.com/)
    - Pull the required models:
      ```bash
      ollama pull llama3
      ollama pull nomic-embed-text
      ```

## Setup

1.  **Clone the repository** (if applicable) or navigate to the project directory.

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables**:
    Create a `.env` file to manage environment variables (e.g., if you switch to WatsonX or OpenAI).
    ```bash
    cp .env.example .env  # or create a file named .env
    ```

## Project Structure

The project is broken down into separate labs for educational purposes:

- `lab1_document_loader.py`: Loads the PDF document using `PyPDFLoader`.
- `lab2_text_splitter.py`: Splits the loaded text into manageable chunks using `RecursiveCharacterTextSplitter`.
- `lab3_embedding.py`: Handles embedding generation. Configured to use **Ollama (nomic-embed-text)** by default, with a fallback to HuggingFace.
- `lab4_vector_db.py`: Orchestrates the loading, splitting, and embedding process, then stores the vectors in a local ChromaDB instance (`./chroma_db`).
- `lab5_retriever.py`: demonstrating how to retrieve relevant documents from the vector database.
- `lab6_gradio.py`: The main application. Sets up the RAG chain and launches the Gradio chat interface.

## Running the Application

### 1. Initialize the Vector Database
Before running the chat app, you must process the PDF and store its embeddings. Run:

```bash
python lab4_vector_db.py
```

This will create a `chroma_db` directory with the vector index.

### 2. Start the Chat Interface
Launch the Gradio app:

```bash
python lab6_gradio.py
```

The application will be available at `http://127.0.0.1:7860`.

## Notes

- **Model Selection**: The code is currently configured to use `llama3` via Ollama. If you wish to use IBM WatsonX or OpenAI, you will need to modify `lab6_gradio.py` and `lab3_embedding.py` and provide the appropriate API keys in `.env`.
- **PDF Source**: The default PDF is `A_Comprehensive_Review_of_Low_Rank_Adaptation_in_Large_Language_Models_for_Efficient_Parameter_Tuning-1.pdf`. To use a different PDF, update the filename in `lab4_vector_db.py`.
