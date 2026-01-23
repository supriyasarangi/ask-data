import uuid
from langchain.memory import ConversationBufferMemory


def create_new_conversation():
    return {
        "messages": [],
        "memory": ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        "knowledge_base": None,
        "pdf_name": None,
        "followups": [],
    }


def init_session_state(st):
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}

    if "current_session_id" not in st.session_state:
        sid = str(uuid.uuid4())
        st.session_state.current_session_id = sid
        st.session_state.conversations[sid] = create_new_conversation()

    if "pending_followup" not in st.session_state:
        st.session_state.pending_followup = None

    return st.session_state.conversations[
        st.session_state.current_session_id
    ]
