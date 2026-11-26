from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

model = ChatOllama(
    model = 'gemma3:1b',
    temperature=0.3
)

chat_history = [
    SystemMessage(content='You are a helpful AI Assistant')
]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == 'exit':
        break
    
    result = model.invoke(chat_history)
    print('AI: ', result.content)
    chat_history.append(AIMessage(content=result.content))
    
print(chat_history)