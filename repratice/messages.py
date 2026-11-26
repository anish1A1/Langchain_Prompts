from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

model = ChatOllama(
    model = 'gemma3:1b',
    temperature = 0
)

messages = [
    SystemMessage(content='You are helpful assistant'),
    HumanMessage(content='Tell me about langchain and how to use RAG with it')
]

result = model.invoke(messages)
print(result.content)
messages.append(AIMessage(content=result.content))
print(messages)