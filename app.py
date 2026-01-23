from dotenv import load_dotenv
import streamlit as st

from conversation import init_session_state
from pdf_utils import build_knowledge_base
from rag_chain import create_rag_chain
from follow_up import generate_followups

load_dotenv()

st.set_page_config(page_title="Ask-Data", page_icon="🤖")
st.header("Ask-Data 🤖💬")


def scroll_to_bottom():
    st.markdown(
        """
        <div id="bottom"></div>
        <script>
            document.getElementById("bottom").scrollIntoView({behavior: "smooth"});
        </script>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Session init
# ----------------------------
current = init_session_state(st)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.subheader("💬 Conversations")

    if st.button("➕ New Chat"):
        from conversation import create_new_conversation
        import uuid

        sid = str(uuid.uuid4())
        st.session_state.current_session_id = sid
        st.session_state.conversations[sid] = create_new_conversation()
        st.rerun()

    st.divider()

    for sid, convo in st.session_state.conversations.items():
        label = convo["pdf_name"] or f"Chat {sid[:6]}"
        if st.button(label, key=f"chat_btn_{sid}"):
            st.session_state.current_session_id = sid
            st.rerun()

# ----------------------------
# PDF upload
# ----------------------------
pdf = st.file_uploader(
    "Upload a PDF for this conversation",
    type="pdf",
    key=f"uploader_{st.session_state.current_session_id}"
)

if pdf and current["knowledge_base"] is None:
    current["pdf_name"] = pdf.name
    current["knowledge_base"] = build_knowledge_base(pdf)

# ----------------------------
# Render chat history
# ----------------------------
for msg in current["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------
# Resolve input
# ----------------------------
# Always show input if knowledge base exists
user_input = None

typed_input = None
if current["knowledge_base"]:
    typed_input = st.chat_input("Ask a question about your PDF...")

if st.session_state.pending_followup:
    user_input = st.session_state.pending_followup
    st.session_state.pending_followup = None
elif typed_input:
    user_input = typed_input

# ----------------------------
# RAG execution
# ----------------------------
if user_input:
    current["messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    llm, chain = create_rag_chain(
        current["knowledge_base"],
        current["memory"]
    )

    result = chain({"question": user_input})
    response = result["answer"]

    with st.chat_message("assistant"):
        st.markdown(response)

    scroll_to_bottom()

    current["messages"].append(
        {"role": "assistant", "content": response}
    )

    try:
        current["followups"] = generate_followups(llm, response)[:3]
    except Exception:
        current["followups"] = []

# ----------------------------
# Follow-ups (after answer)
# ----------------------------
if current.get("followups"):
    st.markdown("**Suggested follow-ups:**")
    cols = st.columns(len(current["followups"]))

    for i, q in enumerate(current["followups"]):
        if cols[i].button(
                q,
                key=f"followup_{st.session_state.current_session_id}_{i}"
        ):
            st.session_state.pending_followup = q
            current["followups"] = []

if not current["knowledge_base"]:
    st.info("📄 Upload a PDF to start chatting")
