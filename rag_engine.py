import faiss
import numpy as np
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Load PDF files
def load_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    mypages = [i.extract_text() or "" for i in reader.pages]
    full_text = "\n".join(mypages).strip()

    if not full_text:
        raise ValueError("No readable text found in PDF")

    return full_text


# Text chunking
def create_chunks(full_text, chunksize=500, overlap=100):
    chunks = []
    start = 0

    while start < len(full_text):
        end = start + chunksize
        chunk = full_text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# Generate embeddings and build FAISS index
def build_vectors(chunks, model_name="all-MiniLM-L6-v2"):

    if not chunks:
        raise ValueError("Cannot apply embedding without chunks")

    embed_model = SentenceTransformer(model_name)

    embed_vectors = embed_model.encode(
        chunks,
        convert_to_numpy=True
    )

    embed_vectors = np.asarray(embed_vectors, dtype="float32")

    faiss.normalize_L2(embed_vectors)

    index = faiss.IndexFlatIP(embed_vectors.shape[1])
    index.add(embed_vectors)

    return index, embed_model


# Load PDFs, create chunks and build vectors
def load_base(pdf_path="railway.docs"):

    chunks = []

    pdf_files = list(Path(pdf_path).glob("*.pdf"))

    if not pdf_files:
        raise ValueError("No PDF files found in railway.docs folder")

    for pdf in pdf_files:
        full_text = load_pdf(pdf)
        chunks.extend(create_chunks(full_text))

    index, embed_model = build_vectors(chunks)

    return index, embed_model, chunks


# Retrieve relevant content
def retrieve_content(question, embed_model, chunks, index, num_chunks=3):

    if not question.strip():
        raise ValueError("Question cannot be blank")

    if num_chunks < 1:
        raise ValueError("Request at least one chunk")

    query_embed = embed_model.encode(
        [question],
        convert_to_numpy=True
    )

    query_embed = np.asarray(query_embed, dtype="float32")

    faiss.normalize_L2(query_embed)

    sim_scores, vec_indices = index.search(
        query_embed,
        min(num_chunks, len(chunks))
    )

    background = ""

    for i in vec_indices[0]:
        if i != -1:
            background += chunks[i] + "\n\n"

    return background


# Generate answer using Groq LLM
def ask_rag(
    question,
    embed_model,
    chunks,
    index,
    api_key,
    num_chunks=3,
    model_name="openai/gpt-oss-120b"
):

    if not api_key:
        raise ValueError("Groq API key missing")

    background = retrieve_content(
        question,
        embed_model,
        chunks,
        index,
        num_chunks=num_chunks
    )

    prompt = f"""
    You are a professional Indian Railway AI assistant.

    Answer the user's question using only the document
    background provided below.

    Do not invent or assume facts.

    If the information is not available in the documents,
    simply say that the information is not available
    in the uploaded documents.

    Document Background:
    {background}

    User Question:
    {question}

    Answer:
    """

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
