import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

# Load Sentence Transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the upgraded QA model and tokenizer
qa_model_name = "google/flan-t5-large"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
qa_model.to(device)

# Streamlit UI
st.title("📖 AI-Powered Question Answering")
st.write("Enter a passage, then ask a question based on it!")
st.write("Please keep the word limit to max 500 only")

# User inputs a paragraph
user_passage = st.text_area("📝 Enter a paragraph:", height=200)

# Always show question input field
user_question = st.text_input("❓ Ask a question based on the paragraph:")

# Function to generate an answer
def generate_answer(question, passage_text):
    """Generate an answer using the QA model."""
    if not passage_text.strip():
        return "⚠️ Please enter a passage first."

    input_text = f"question: {question}  context: {passage_text}"
    input_ids = qa_tokenizer.encode(input_text, return_tensors="pt").to(device)

    # Generate answer with increased max_length
    output_ids = qa_model.generate(input_ids, max_length=200)

    return qa_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Display answer only when both inputs are provided
if user_passage and user_question:
    st.write("🔍 Generating answer...")
    answer = generate_answer(user_question, user_passage)
    
    st.subheader("💡 Answer:")
    st.write(answer)
elif user_question:
    st.warning("❗ Please enter a passage before asking a question.")
