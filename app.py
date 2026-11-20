import streamlit as st
from rag_engine import load_base, ask_rag

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="Indian Railway Chatbot",
    page_icon="🚆",
    layout="centered"
)

# ---------------- SIDEBAR ----------------

st.sidebar.title("🚆 Indian Railway RAG")

st.sidebar.markdown("---")

st.sidebar.subheader("📌 About the Project")
st.sidebar.write(
    "Indian Railway Chatbot is a Retrieval Augmented Generation (RAG) "
    "application that answers questions using information from railway PDF documents."
)

st.sidebar.markdown("---")

st.sidebar.subheader("🛠️ Technologies Used")
st.sidebar.markdown("""
- Python
- Streamlit
- Retrieval Augmented Generation (RAG)
- PyPDF
- Sentence Transformers
- FAISS
- Groq LLM
""")

st.sidebar.markdown("---")

st.sidebar.subheader("🔄 RAG Pipeline")

st.sidebar.markdown("""
📄 Railway PDF Documents

⬇️

📖 Text Extraction

⬇️

✂️ Text Chunking

⬇️

🧠 Embedding Generation

⬇️

🗄️ FAISS Vector Database

⬇️

🔍 Relevant Information Retrieval

⬇️

🤖 Groq LLM

⬇️

💬 Final Answer
""")

st.sidebar.markdown("---")

st.sidebar.caption("Developed by Team | Indian Railway RAG Project")

# ---------------- MAIN PAGE ----------------

st.title("🚆 Indian Railway Chatbot")

st.caption("Your AI-powered assistant for Indian Railway information.")

st.caption("Ask questions about Indian Railways, its history, Vande Bharat Express and railway stations.")

st.info("💡 Ask a question to get an answer based on the uploaded railway documents.")

# ---------------- API KEY ----------------

my_key = st.secrets["GROQ_API_KEY"]

# ---------------- LOAD RAG ENGINE ----------------

@st.cache_resource
def initiate_rag():
    return load_base("railway.docs")

index, embed_model, chunks = initiate_rag()

# ---------------- CHAT HISTORY ----------------

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

for qa in st.session_state.chat_hist:
    with st.chat_message(qa["role"]):
        st.write(qa["content"])

# ---------------- CHAT INPUT ----------------

question = st.chat_input("🚆 Ask anything about Indian Railways...")

if question:

    st.session_state.chat_hist.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("🔍 Searching railway documents and generating answer..."):

            answer = ask_rag(
                question,
                embed_model,
                chunks,
                index,
                api_key=my_key
            )

            st.write(answer)

            st.session_state.chat_hist.append(
                {"role": "assistant", "content": answer}
            )

# ---------------- CLEAR CHAT ----------------

if st.button("🗑️ Clear Chat"):
    st.session_state.chat_hist = []
    st.rerun()
