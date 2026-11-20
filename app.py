import streamlit as st
from rag_engine import load_base, ask_rag

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Indian Railway Chatbot",
    page_icon="🚆",
    layout="wide"
)

# --------------------------------------------------
# MAIN TITLE
# --------------------------------------------------

st.title("🚆 Indian Railway Chatbot")

st.caption(
    "Your AI-powered assistant for information about Indian Railways."
)

st.write(
    "Ask questions about Indian Railways, its history, "
    "Vande Bharat Express, railway stations and more."
)

# --------------------------------------------------
# SIDEBAR - PROJECT INFORMATION
# --------------------------------------------------

st.sidebar.title("🚆 Railway RAG Assistant")

with st.sidebar.expander("ℹ️ About Project", expanded=True):

    st.write(
        """
        Indian Railway Chatbot is an AI-powered chatbot
        developed using Retrieval-Augmented Generation (RAG).

        It retrieves relevant information from railway PDF
        documents and generates answers using a Large
        Language Model.
        """
    )

# --------------------------------------------------
# TECHNOLOGIES USED
# --------------------------------------------------

with st.sidebar.expander("🛠️ Technologies Used"):

    st.markdown(
        """
        - Python
        - Streamlit
        - Retrieval-Augmented Generation (RAG)
        - PyPDF
        - Sentence Transformers
        - FAISS Vector Database
        - Groq LLM
        - NumPy
        - Graphviz
        """
    )

# --------------------------------------------------
# GRAPHICAL RAG PIPELINE
# --------------------------------------------------

with st.sidebar.expander("🔄 RAG Pipeline Flowchart", expanded=True):

    rag_flowchart = """
    digraph RAG {

        rankdir=TB;

        node [
            shape=box,
            style="rounded,filled",
            color="#2874A6",
            fillcolor="#D6EAF8",
            fontname="Arial"
        ];

        edge [
            color="#34495E"
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

    st.graphviz_chart(rag_flowchart, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Powered by Retrieval-Augmented Generation")

# --------------------------------------------------
# GROQ API KEY
# --------------------------------------------------

try:

    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

except Exception:

    st.error(
        "Groq API key not found. Please add GROQ_API_KEY "
        "in Streamlit Community Cloud Secrets."
    )

    st.stop()

# --------------------------------------------------
# LOAD RAG KNOWLEDGE BASE
# --------------------------------------------------

@st.cache_resource
def initiate_rag():

    return load_base("railway.docs")


try:

    with st.spinner("Loading railway knowledge base..."):

        index, embed_model, chunks = initiate_rag()

except Exception as e:

    st.error(f"Error loading railway documents: {e}")

    st.stop()

# --------------------------------------------------
# INITIALIZE CHAT HISTORY
# --------------------------------------------------

if "chat_hist" not in st.session_state:

    st.session_state.chat_hist = []

# --------------------------------------------------
# CLEAR CHAT BUTTON
# --------------------------------------------------

if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):

    st.session_state.chat_hist = []

    st.rerun()

# --------------------------------------------------
# DISPLAY CHAT WINDOW
# --------------------------------------------------

st.subheader("💬 Chat with Indian Railway Assistant")

for message in st.session_state.chat_hist:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# --------------------------------------------------
# USER CHAT INPUT
# --------------------------------------------------

user_question = st.chat_input(
    "Ask something about Indian Railways..."
)

if user_question:

    # Display user question

    with st.chat_message("user"):

        st.markdown(user_question)

    # Save user question

    st.session_state.chat_hist.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Generate assistant response

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching railway documents and generating answer..."
        ):

            try:

                answer = ask_rag(
                    user_question,
                    embed_model,
                    chunks,
                    index,
                    GROQ_API_KEY
                )

                st.markdown(answer)

            except Exception as e:

                answer = f"Sorry, an error occurred: {e}"

                st.error(answer)

    # Save assistant response

    st.session_state.chat_hist.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# --------------------------------------------------
# CHAT HISTORY IN SIDEBAR
# --------------------------------------------------

st.sidebar.markdown("---")

with st.sidebar.expander("💬 Chat History", expanded=True):

    if st.session_state.chat_hist:

        for message in st.session_state.chat_hist:

            if message["role"] == "user":

                st.markdown(
                    f"**👤 You:** {message['content']}"
                )

            else:

                st.markdown(
                    f"**🤖 Assistant:** {message['content']}"
                )

            st.divider()

    else:

        st.caption(
            "Your conversation history will appear here."
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    "<center>🚆 Indian Railway Chatbot | "
    "Built using Python, Streamlit and RAG</center>",
    unsafe_allow_html=True
)
