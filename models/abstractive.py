from transformers import pipeline
import multiprocessing
from functools import partial

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def process_chunk(chunk, summarizer, max_length, min_length):
    summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def abstractive_summarize(input_text, num_sentences=3):
    max_input_length = 1024
    chunk_size = max_input_length // 2
    words = input_text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        summarizer_partial = partial(process_chunk, summarizer=summarizer, max_length=150, min_length=num_sentences * 5)
        chunk_summaries = pool.map(summarizer_partial, chunks)

    final_summary = " ".join(chunk_summaries)
    return final_summary
