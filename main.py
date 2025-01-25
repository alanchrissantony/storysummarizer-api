from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch


app = FastAPI()


nlp = spacy.load("en_core_web_sm")


MODEL_NAME = "facebook/bart-large-cnn"  
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


class SummarizationRequest(BaseModel):
    text: str
    method: str  
    max_length: int = 50  



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



def generate_intro(text, max_length=50):
    prompt = f"Write a short description for the story: {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=5,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def generate_abstractive_summary(text, max_length=150):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        num_beams=5,
        length_penalty=2.0,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# API Route
@app.post("/summarize/")
def summarize(request: SummarizationRequest):
    text = request.text
    method = request.method
    max_length = request.max_length

    if not text.strip():
        return {"error": "Text cannot be empty."}

    if method == "extractive":
        summary = summarize_extractive(text)
    elif method == "abstractive":
        summary = generate_intro(text, max_length)
    else:
        return {"error": f"Unsupported method: {method}"}

    return {"summary": summary}
