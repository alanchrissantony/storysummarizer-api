import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import multiprocessing
from functools import partial
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load SpaCy model for extractive summarization
nlp = spacy.load("en_core_web_sm")

# Load Llama-3.2-3B-Instruct-4bit model for abstractive summarization
MODEL_NAME = "Llama-3.2-3B-Instruct-4bit"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


# Functions for extractive summarization
def get_sentence_embeddings(doc):
    return np.array([sent.vector for sent in doc.sents])


def build_similarity_matrix(sentences, embeddings):
    sim_matrix = cosine_similarity(embeddings, embeddings)
    np.fill_diagonal(sim_matrix, 0)  # Remove self-loops
    return sim_matrix


def apply_textrank(sim_matrix, sentences, damping=0.85, max_iter=100, tol=1e-4):
    n = len(sentences)
    scores = np.ones(n) / n
    for _ in range(max_iter):
        new_scores = (1 - damping) / n + damping * sim_matrix.T @ scores
        if np.linalg.norm(new_scores - scores) <= tol:
            break
        scores = new_scores
    return scores


def summarize_extractive(text, num_sentences=3):
    doc = nlp(text)
    sentences = list(doc.sents)
    embeddings = get_sentence_embeddings(doc)
    sim_matrix = build_similarity_matrix(sentences, embeddings)
    scores = apply_textrank(sim_matrix, sentences)
    ranked_sentences = [sentences[i].text for i in np.argsort(scores)[::-1]]
    return " ".join(ranked_sentences[:num_sentences])


# Abstractive summarization using Llama and multiprocessing
def process_chunk(chunk, summarizer, max_length, min_length):
    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def abstractive_summarize(input_text, num_sentences=3):
    max_input_length = 1024  # Token limit
    chunk_size = max_input_length // 2
    words = input_text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Custom prompt to enhance summaries
    prompt = "Summarize the following story in a clear, concise, and engaging manner. Capture the main themes, key events, and any important characters or details, while ensuring the summary is easy to understand and preserves the essence of the original narrative. "

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        summarizer_partial = partial(
            process_chunk,
            summarizer=summarizer,
            max_length=150,
            min_length=num_sentences * 5
        )
        chunk_summaries = pool.map(
            lambda chunk: summarizer(f"{prompt} {chunk}", max_length=150, min_length=num_sentences * 5, do_sample=False)[0]['summary_text'],
            chunks
        )

    final_summary = " ".join(chunk_summaries)
    return final_summary


# Streamlit App
st.title("Text Summarizer with Llama-3.2-3B-Instruct-4bit")

# Input text
input_text = st.text_area("Enter the text you want to summarize:", height=200)

# Select summarization method
method = st.radio(
    "Select summarization method:",
    ("Extractive", "Abstractive")
)

# Adjust maximum length for abstractive summaries
num_sentences = st.slider("Number of sentences for summary:", 1, 10, 3)

# Generate summary
if st.button("Summarize"):
    if not input_text.strip():
        st.error("Please enter some text.")
    else:
        if method == "Extractive":
            summary = summarize_extractive(input_text, num_sentences)
        else:
            summary = abstractive_summarize(input_text, num_sentences)

        st.subheader("Summary:")
        st.write(summary)