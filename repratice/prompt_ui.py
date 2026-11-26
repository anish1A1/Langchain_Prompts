from langchain_ollama import ChatOllama
import streamlit as st

model = ChatOllama(
    model="phi4-mini",
    temperature=0,
)

st.header('Reasearch Tool')

user_input = st.text_input("Enter your Prompt: ")

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)
    print(result.content)