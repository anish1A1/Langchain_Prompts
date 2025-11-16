from langchain_core.prompts import ChatPromptTemplate

# We need to provide the texts in tupple as shown below when using ChatPromptTemplate

chat_templates = ChatPromptTemplate([
    {'system', 'You are a helpful {domain} expert'},
    {'human', 'Explain in simple terms, what is {topic}'}
])

prompt = chat_templates.invoke({'domain': 'cricket', 'topic': 'Dursa'})
print(prompt)