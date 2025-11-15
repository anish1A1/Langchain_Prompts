from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)


"""
# This is a simple chatbot, the problem with this is it doesnot store previous messages and hallusinate if asked previous message inputs.

while True:
    user_input = input("You: ")
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    print("AI: ", result.content)
    
"""

# Now to get the chatbot to know previous chat history with the present
# We can create a list which stores messages (both AI and human).

chat_history = [
    SystemMessage(content='You are a helpful AI Assistant')
]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    
    if user_input == 'exit':
        break
    result = model.invoke(chat_history)
    # Added chat_history as it contains present as well as past messages.
    
    chat_history.append(AIMessage(content=result.content)) #storing the present AI message in history
    print("AI: ", result.content)
    
print(chat_history) 
    