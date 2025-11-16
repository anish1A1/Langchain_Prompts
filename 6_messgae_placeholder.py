from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='6_chat_history'),
    ('human', '{query}')
])
# We need to place MessagePlaceholder in middle as it will get previous msg history before the human msg input.

# A list to store all the present and previous msg, and provide to the model.
chat_history = []

# load chat history

with open('6_chat_history.txt') as f:
    chat_history.extend(f.readlines())
# added the 6_chat_history.txt msges in the list (chat-history)

print(chat_history)


result = chat_template.invoke({
    '6_chat_history' : chat_history,
    'query': 'Where is my refund?'
})

print(result)