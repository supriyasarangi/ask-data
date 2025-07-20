
from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI  # <-- use OpenAI-compatible LLM

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with data")
    st.header("Chat with your data 🤖💬")

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = HuggingFaceEmbeddings()

        # use Chroma vector store
        knowledge_base = Chroma.from_texts(
            chunks,
            embeddings,
            persist_directory="chroma_db"
        )
        knowledge_base.persist()

        # show user input
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            # ✅ Use Groq-hosted LLaMA3 or Mixtral via OpenAI-compatible API
            llm = ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
                model="llama3-70b-8192",  # or "mixtral-8x7b-32768", "gemma-7b-it"
                temperature=0
            )

            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.write(response)

if __name__ == '__main__':
    main()