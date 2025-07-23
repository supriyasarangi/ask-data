from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

load_dotenv()


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="🤖")
    st.header("Chat with your PDF 🤖💬")

    # Upload PDF
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None

    # Sidebar: Toggle to view conversation history
    with st.sidebar:
        with st.expander("🕘 Conversation History", expanded=False):
            for i, (q, a) in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")
    
    # Extract and process PDF
    if pdf is not None and st.session_state.knowledge_base is None:
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings()
        knowledge_base = Chroma.from_texts(
            chunks, embeddings, persist_directory="chroma_db"
        )
        knowledge_base.persist()
        st.session_state.knowledge_base = knowledge_base

    # Chat UI
    if st.session_state.knowledge_base:
        user_input = st.chat_input("Ask a question about your PDF...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)

            docs = st.session_state.knowledge_base.similarity_search(user_input)

            # Use Groq-hosted LLM
            llm = ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama3-70b-8192",
                temperature=0,
            )
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_input)

            with st.chat_message("assistant"):
                st.markdown(response)

            # Save to session history
            st.session_state.chat_history.append((user_input, response))


if __name__ == "__main__":
    main()
