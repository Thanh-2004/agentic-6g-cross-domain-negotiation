import json
import logging

from typing import List, Any, Dict

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models import BaseChatModel

class MockPart:
    def __init__(self, text: str):
        self.text = text
        self.function_call = None


class MockContent:
    def __init__(self, text: str):
        self.parts = [MockPart(text)]

class MockCandidate:
    def __init__(self, text: str):
        self.content = MockContent(text)

class MockResponse:
    def __init__(self, text):
        self.text = ext
        self.candidates = [MockCandidate(text)]

class LangchainSessionAdapter:
    def __init__(self, llm: BaseChatModel, history: List = None):
        self.llm = llm
        ## Simple chat history
        self.history = history if history else []

    def send_message(self, content: Any) -> MockResponse:
        '''
        Alternative function for chat_session.senf_message
        :param content: input for LLM
        :return: LLM message
        '''

        prompt_text = ""

        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    prompt_text += part["text"] + "\n"
                elif hasattr(part, "text"):
                    prompt_text += part.text + "\n"
                else:
                    prompt_text += str(part) + "\n"
        else:
            prompt_text = str(content)

        messages = self.history + HumanMessage(content=prompt_text)

        try:
            ai_message = self.llm.invoke(message)
            response_text = ai_message.content
        except Exception as e:
            logging.error(f"LangChain invoke failed: {e}")
            response_text = "NO_AGREEMEN_POSSIBLE: Error calling external LLM"

        self.history.append(HumanMessage(content=prompt_text))
        self.history.append(AIMessage(content=response_text))

        return MockResponse(response_text)



class LLMWrapper:
    def __init__(self, langchain_llm: BaseChatModel):
        self.llm = langchain_llm

    def start_chat(self, history: List = None):
        return LangchainSessionAdapter(self.llm, history)



# def llm_wrapping(original_agent_class, langchain_llm, *args, **kwargs):
#     '''
#     Create an instance of agent with GoogleGenAI functions
#     :param original_agent_class:
#     :param langchain_llm:
#     :param args:
#     :param kwargs:
#     :return:
#     '''
#
#     ## wrap agent into Google functions
#     wrapper = LLMWrapper(langchain_llm)
#     agent.model = wrapper
#     agent.chat_session = wrapper.start_chat()
#
#     logging.info(f"[{agent.name}] Successfully wrapped with externalLLM: {type(langchain_llm).__name__}")
#
#     return agent
#



