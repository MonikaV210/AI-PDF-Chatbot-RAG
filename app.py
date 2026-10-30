
import streamlit as st
from pypdf import PdfReader
import ollama

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Create the chatbot webpage
st.title("AI PDF Chatbot")
st.write("Welcome to my AI PDF Assistant!")

# 2. Upload the PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

text = ""

if uploaded_file is not None:
    reader = PdfReader(uploaded_file)

    for page in reader.pages:
        text += page.extract_text() or ""

    if text.strip():
        st.success("PDF uploaded and read successfully!")

        with st.expander("View Extracted PDF Text"):
            st.write(text)
    else:
        st.warning("No readable text found in this PDF.")


# 3. Divide the PDF text into chunks
def split_text(text, chunk_size=1000, overlap=150):
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# 4. Find relevant chunks using TF-IDF
def find_relevant_chunks(question, chunks, top_k=3):
    vectorizer = TfidfVectorizer(stop_words="english")

    vectors = vectorizer.fit_transform(chunks + [question])

    similarities = cosine_similarity(
        vectors[-1], vectors[:-1]
    ).flatten()

    best_indices = similarities.argsort()[::-1][:top_k]

    relevant_chunks = [
        chunks[i] for i in best_indices if similarities[i] > 0
    ]

    return relevant_chunks


# 5. Ask a question
question = st.text_input("Ask a question about the PDF")

if question:
    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif not text.strip():
        st.warning("The uploaded PDF contains no readable text.")

    else:
        chunks = split_text(text)

        relevant_chunks = find_relevant_chunks(question, chunks)

        if not relevant_chunks:
            st.info("The answer is not mentioned in the PDF.")

        else:
            context = "\n\n".join(relevant_chunks)

            # Show retrieved text for debugging
            with st.expander("View Retrieved PDF Chunks"):
                st.write(context)

            # 6. Generate answer using local Qwen model
            with st.spinner("AI is thinking..."):
                try:
                    response = ollama.chat(
                        model="qwen2.5:3b",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a PDF question-answering assistant. "
                                    "Answer using only the provided PDF context. "
                                    "If the context does not contain the answer, "
                                    "say exactly: "
                                    "'The answer is not mentioned in the PDF.' "
                                    "Do not guess or invent facts. "
                                    "Keep answers short and clear."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"PDF Context:\n{context}\n\n"
                                    f"Question: {question}"
                                )
                            }
                        ]
                    )

                    st.subheader("AI Answer")
                    st.write(response["message"]["content"])

                except Exception as e:
                    st.error(f"Could not connect to Ollama: {e}")
 