from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

# We state messages according to the type of user because when working in big project we need to store the messages according to the user type. for e.g which msg was written by ai like with the user.

llm = HuggingFaceEndpoint(
    repo_id= "moonshotai/Kimi-K2-Thinking",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content='You are a helpful assistant'),
    HumanMessage(content='Tell me about Langchain')
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))
print(messages)