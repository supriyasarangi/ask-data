from dotenv import load_dotenv
import os
import uuid
import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

load_dotenv()


def create_new_conversation():
    return {
        "messages": [],
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        "knowledge_base": None,
        "pdf_name": None,
    }


def main():
    st.set_page_config(page_title="Chat with PDF", page_icon="🤖")
    st.header("Chat with your PDF 🤖💬")

    # ----------------------------
    # Initialize conversations
    # ----------------------------
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if "current_session_id" not in st.session_state:
        session_id = str(uuid.uuid4())
        st.session_state.current_session_id = session_id
        st.session_state.conversations[session_id] = create_new_conversation()

    current = st.session_state.conversations[
        st.session_state.current_session_id
    ]

    # ----------------------------
    # Sidebar: conversation switch
    # ----------------------------
    with st.sidebar:
        st.subheader("💬 Conversations")

        if st.button("➕ New Chat"):
            session_id = str(uuid.uuid4())
            st.session_state.current_session_id = session_id
            st.session_state.conversations[session_id] = create_new_conversation()
            st.rerun()

        st.divider()

        for sid, convo in st.session_state.conversations.items():
            label = convo["pdf_name"] or f"Chat {sid[:6]}"
            if st.button(label, key=f"chat_btn_{sid}"):
                st.session_state.current_session_id = sid
                st.rerun()

    # ----------------------------
    # PDF upload (per conversation)
    # ----------------------------
    pdf = st.file_uploader(
        "Upload a PDF for this conversation",
        type="pdf",
        key=f"uploader_{st.session_state.current_session_id}"
    )

    if pdf is not None and current["knowledge_base"] is None:
        current["pdf_name"] = pdf.name

        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings()
        current["knowledge_base"] = Chroma.from_texts(chunks, embeddings)

    # ----------------------------
    # Render chat history
    # ----------------------------
    for msg in current["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ----------------------------
    # Chat input + RAG
    # ----------------------------
    if current["knowledge_base"]:
        user_input = st.chat_input("Ask a question about your PDF...")

        if user_input:
            current["messages"].append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            llm = ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama-3.1-8b-instant",
                temperature=0,
            )

            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=current["knowledge_base"].as_retriever(),
                memory=current["memory"]
            )

            result = qa_chain({"question": user_input})
            response = result["answer"]

            with st.chat_message("assistant"):
                st.markdown(response)

            current["messages"].append(
                {"role": "assistant", "content": response}
            )

    else:
        st.info("📄 Upload a PDF to start chatting")


if __name__ == "__main__":
    main()
