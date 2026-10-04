import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model="gemini-3.8-flash",
    api_key=os.getenv("GEMINI_API_KEY"),  # uses GOOGLE_API_KEY
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant. Use the conversation history to stay consistent."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])
chain = prompt | model

history = []  # starts EMPTY — nothing hardcoded

while True:  # one loop = one chat turn
    question = input("You: ")
    if question.strip().lower() in {"quit", "exit"}:
        break

    answer = chain.invoke(
        {"history": history, "question": question}).content  # send history so far
    print("Bot:", answer)

    history.append(HumanMessage(question))  # remember what you said...
    history.append(AIMessage(answer))  # ...and what it replied
    print(f"   (history now has {len(history)} messages)")


