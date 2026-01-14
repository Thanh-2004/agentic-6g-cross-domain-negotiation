from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model = "llama3.2:1b",
    temperature=0.7
)

response = llm.invoke("Hi, who are you?")

print(response.content)
