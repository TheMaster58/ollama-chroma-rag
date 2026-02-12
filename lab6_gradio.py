import gradio as gr
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama


from lab5_retriever import get_retriever
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    print("Using ChatOllama (llama3)")
    return ChatOllama(model="llama3", temperature=0)


def qa_bot(message, history):
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
    demo = gr.ChatInterface(
        fn=qa_bot,
        title="RAG with Gradio and Ollama",
        description="Ask questions about the Low-Rank Adaptation (LoRA) paper. Ensure you have 'ollama serve' running and 'llama3' pulled.",
        examples=["What is LoRA?", "How does LoRA compare to fine-tuning?", "What are the benefits of LoRA?"],
    )
    
    try:
        retriever = get_retriever() # Check if vector/retriever is ready
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except Exception as e:
        print(f"Failed to start: {e}")
