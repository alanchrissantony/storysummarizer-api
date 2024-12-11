import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


nlp = spacy.load("en_core_web_sm")

def get_sentence_embeddings(doc):
    return np.array([sent.vector for sent in doc.sents])

def build_similarity_matrix(sentences, embeddings):
    sim_matrix = cosine_similarity(embeddings, embeddings)
    np.fill_diagonal(sim_matrix, 0)
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

def extractive_summarize(input_text, num_sentences=3):
    doc = nlp(input_text)
    sentences = list(doc.sents)
    embeddings = get_sentence_embeddings(doc)
    sim_matrix = build_similarity_matrix(sentences, embeddings)
    scores = apply_textrank(sim_matrix, sentences)
    ranked_sentences = [sentences[i].text for i in np.argsort(scores)[::-1]]
    return " ".join(ranked_sentences[:num_sentences])