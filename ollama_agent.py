from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

llm = ChatOllama(
    model = "gpt-oss:20b",
    temperature=0.7
)

response = llm.invoke("Hi, who are you?")

print(response.content)
