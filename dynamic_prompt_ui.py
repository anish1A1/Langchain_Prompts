from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, load_prompt

import streamlit as st 
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "HuggingFaceH4/zephyr-7b-beta",
    task="text-generation"
)


model = ChatHuggingFace(llm=llm)

st.header('Reasearch Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


# Used a resuable template 
template = load_prompt('template.json')


# fills the placeholders 
prompt = template.invoke({
    'paper_input': paper_input,
    'style_input': style_input,
    'length_input': length_input
})

if st.button("Summarize"):
    result = model.invoke(prompt)
    print(result.content)
    st.write(result.content)