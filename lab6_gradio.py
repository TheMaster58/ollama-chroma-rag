import gradio as gr
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma # Added

from lab1_document_loader import load_document # Added
from lab2_text_splitter import split_text # Added
from lab3_embedding import get_embedding_model # Added
from lab5_retriever import get_retriever
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    # print("Using ChatOllama (llama3)") # Reduced noise
    return ChatOllama(model="gpt-oss:20b", temperature=0)

def process_pdf(file):
    if not file:
        return "No file uploaded."
    
    try:
        # 1. Load
        print(f"Processing uploaded file: {file.name}")
        docs = load_document(file.name)
        
        # 2. Split
        chunks = split_text(docs)
        
        # 3. Embed and Store (Append)
        embedding_model = get_embedding_model()
        persist_directory = 'chroma_db'
        
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vectordb.persist()
        
        return f"Processed {os.path.basename(file.name)} successfully! Added {len(chunks)} chunks to the database."
    except Exception as e:
        return f"Error processing file: {str(e)}"

def qa_bot_logic(message, history):
    try:
        # 1. Get Retriever
        retriever = get_retriever(k=3)
        
        # 2. Get LLM
        try:
            llm = get_llm()
        except Exception as e:
            return f"Error initializing LLM: {str(e)}\n\nPlease ensure you have 'ollama serve' running and 'llama3' model pulled."

        # 3. Create Retrieval Chain
        prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

        document_chain = create_stuff_documents_chain(llm, prompt)
        qa = create_retrieval_chain(retriever, document_chain)
        
        # 4. Run Query
        response = qa.invoke({"input": message})
        return response['answer']
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Setup Gradio Interface
if __name__ == "__main__":
    print("Starting Gradio Server...")
    
    with gr.Blocks(title="RAG with Gradio and Ollama") as demo:
        gr.Markdown("# RAG with Gradio and Ollama")
        gr.Markdown("Ask questions about your documents. Upload a PDF to add it to the knowledge base.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Add Document")
                file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
                process_btn = gr.Button("Process PDF", variant="primary")
                status_output = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column(scale=4):
                gr.Markdown("### Chat")
                chatbot = gr.ChatInterface(
                    fn=qa_bot_logic,
                    examples=["What is LoRA?", "How does LoRA compare to fine-tuning?", "What are the benefits of LoRA?"],
                )

        process_btn.click(process_pdf, inputs=file_input, outputs=status_output)

    try:
        # Check if vector/retriever is ready (optional, just to warn)
        try:
            get_retriever()
        except:
            print("Vector DB not found. Please upload a PDF to initialize.")
            
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except Exception as e:
        print(f"Failed to start: {e}")
