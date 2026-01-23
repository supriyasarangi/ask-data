import os
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain


def create_rag_chain(knowledge_base, memory):
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=knowledge_base.as_retriever(),
        memory=memory
    )

    return llm, chain
