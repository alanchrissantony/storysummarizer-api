import streamlit as st
import requests

# API endpoint
API_URL = "http://127.0.0.1:8000/summarize/"

st.title("Story Summarizer")
st.subheader("Choose Extractive or Abstractive Summarization")

# Input text
text_input = st.text_area("Enter the story text:")
method = st.selectbox("Summarization Type", ["extractive", "abstractive (Intro/Description)"])
max_length = st.slider("Max Length (For Abstractive Intro)", min_value=50, max_value=300, value=50)

# Map method to API-compatible key
method_key = "extractive" if method == "extractive" else "abstractive"

if st.button("Summarize"):
    if text_input.strip():
        # Send request to API
        response = requests.post(API_URL, json={
            "text": text_input,
            "method": method_key,
            "max_length": max_length
        })
        if response.status_code == 200:
            summary = response.json().get("summary", "No summary generated.")
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.error(f"Error: {response.status_code}")
    else:
        st.warning("Please enter some text to summarize.")
