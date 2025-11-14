from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

import streamlit as st 
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

st.header('Research Tool')
