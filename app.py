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

# Project Information
st.sidebar.subheader("📌 About the Project")

st.sidebar.write(
    """
    Indian Railway Chatbot is an AI-powered application
    developed using Retrieval Augmented Generation (RAG).

    It helps users find information about Indian Railways
    using the uploaded railway PDF documents.
    """
)

st.sidebar.markdown("---")

# Technologies Used
st.sidebar.subheader("🛠️ Technologies Used")

st.sidebar.markdown(
    """
    - 🐍 Python
    - 🌐 Streamlit
    - 📄 PyPDF
    - 🧠 Sentence Transformers
    - 🗄️ FAISS Vector Database
    - 🤖 Groq LLM
    - 🔎 Retrieval Augmented Generation
    """
)

st.sidebar.markdown("---")

# RAG Pipeline Flowchart
st.sidebar.subheader("🔄 RAG Pipeline Flowchart")

rag_flowchart = """
digraph RAG {

    graph [
        rankdir=TB,
        bgcolor="transparent",
        nodesep=0.35,
        ranksep=0.4
    ];

    node [
        shape=box,
        style="rounded,filled",
        fillcolor="#DCEEFF",
        color="#2874A6",
        fontname="Arial",
        fontsize=10,
        margin=0.15
    ];

    edge [
        color="#555555",
        arrowsize=0.8
    ];

    A [label="Railway PDF Documents"];
    B [label="Text Extraction"];
    C [label="Text Chunking"];
    D [label="Embedding Generation"];
    E [label="FAISS Vector Database"];

    F [label="User Question"];
    G [label="Question Embedding"];
    H [label="Similarity Search"];
    I [label="Relevant Context"];
    J [label="Groq LLM"];
    K [label="Final Answer"];

    A -> B;
    B -> C;
    C -> D;
    D -> E;

    F -> G;
    G -> H;
    E -> H;

    H -> I;
    I -> J;
    J -> K;
}
"""

st.sidebar.graphviz_chart(rag_flowchart, use_container_width=True)

st.sidebar.markdown("---")

st.sidebar.caption(
    "🚆 Indian Railway Chatbot using RAG"
)

st.sidebar.caption("Developed by Team Indian Railway RAG")

# ---------------- MAIN PAGE ----------------

st.title("🚆 Indian Railway Chatbot")

st.caption(
    "Your AI-powered assistant for Indian Railway information."
)

st.caption(
    "Ask questions about Indian Railways, its history, "
    "Vande Bharat Express and railway stations."
)

st.info(
    "💡 Ask a question to get answers based on the uploaded railway documents."
)

# ---------------- API KEY ----------------

try:
    my_key = st.secrets["GROQ_API_KEY"]
except Exception:
    st.error(
        "Groq API key is missing. Please configure "
        "GROQ_API_KEY in Streamlit Secrets."
    )
    st.stop()

# ---------------- LOAD RAG ENGINE ----------------

@st.cache_resource
def initiate_rag():
    return load_base("railway.docs")


try:
    index, embed_model, chunks = initiate_rag()

except Exception as e:
    st.error(f"Error loading railway documents: {e}")
    st.stop()

# ---------------- CHAT HISTORY ----------------

if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []

# Display previous messages
for qa in st.session_state.chat_hist:

    with st.chat_message(qa["role"]):
        st.write(qa["content"])

# ---------------- CHAT INPUT ----------------

question = st.chat_input(
    "🚆 Ask anything about Indian Railways..."
)

if question:

    # Display user question
    st.session_state.chat_hist.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)

    # Generate AI answer
    with st.chat_message("assistant"):

        with st.spinner(
            "🔍 Searching railway documents and generating answer..."
        ):

            try:
                answer = ask_rag(
                    question,
                    embed_model,
                    chunks,
                    index,
                    api_key=my_key
                )

                st.write(answer)

                st.session_state.chat_hist.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:
                st.error(f"Unable to generate answer: {e}")

# ---------------- CLEAR CHAT ----------------

st.markdown("---")

if st.button("🗑️ Clear Chat", use_container_width=True):

    st.session_state.chat_hist = []

    st.rerun()

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "Indian Railway Chatbot | Retrieval Augmented Generation (RAG)"
)
