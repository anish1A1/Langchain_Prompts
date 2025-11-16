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

user_input = st.text_input("Enter your prompt: ")

if st.button('Summarize'):
    result = model.invoke(user_input)
    # The actual content from chatmodel to display
    st.write(result.content)
    print(result.content)