import streamlit as st
import time
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from google import genai


# Streamlit UI Setup
st.title("RAG-based Video Transcription & QA System")
api_key = st.text_input("Enter your Google API Key:", type="password")
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
video_file = st.file_uploader("Upload a Video File:", type=["mp4"])

if api_key and openai_api_key and video_file:
    client = genai.Client(api_key=api_key)
    st.write("Uploading file...")
    
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    
    uploaded_file = client.files.upload(file="temp_video.mp4")
    st.success(f"Completed upload: {uploaded_file.uri}")

    # Wait for processing
    while uploaded_file.state.name == "PROCESSING":
        time.sleep(1)
        uploaded_file = client.files.get(name=uploaded_file.name)
    
    if uploaded_file.state.name == "FAILED":
        st.error("File processing failed")
        st.stop()

    st.success("File processing done")

    # Transcription
    prompt = "Transcribe the video in detail and extract text from the whiteboard."
    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=[uploaded_file, prompt]
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_text(response.text)

    # Embedding & Vector Storage
    embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vector_store = FAISS.from_texts(splits, embedding_model)

    # QA System
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )

    # Query Input
    query = st.text_input("Ask a question about the video:")
    if query:
        response = qa_chain.run(query)
        st.write("Response:", response)
