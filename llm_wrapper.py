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
        self.text = text
        self.candidates = [MockCandidate(text)]

class LangchainSessionAdapter:
    def __init__(self, llm: BaseChatModel, history: List = None):
        self.llm = llm
        ## Simple chat history
        self.history = history if history else []

    # def send_message(self, content: Any) -> MockResponse:
    #     '''
    #     Alternative function for chat_session.senf_message
    #     :param content: input for LLM
    #     :return: LLM message
    #     '''
    #
    #     prompt_text = ""
    #
    #     if isinstance(content, list):
    #         for part in content:
    #             if isinstance(part, dict) and "text" in part:
    #                 prompt_text += part["text"] + "\n"
    #             elif hasattr(part, "text"):
    #                 prompt_text += part.text + "\n"
    #             else:
    #                 prompt_text += str(part) + "\n"
    #     else:
    #         prompt_text = str(content)
    #
    #     messages = self.history + [HumanMessage(content=prompt_text)]
    #
    #     try:
    #         ai_message = self.llm.invoke(messages)
    #         response_text = ai_message.content
    #     except Exception as e:
    #         logging.error(f"LangChain invoke failed: {e}")
    #         response_text = "NO_AGREEMEN_POSSIBLE: Error calling external LLM"
    #
    #     self.history.append(HumanMessage(content=prompt_text))
    #     self.history.append(AIMessage(content=response_text))
    #
    #     return MockResponse(response_text)

    def send_message(self, content: Any) -> MockResponse:
        '''
        Hàm thay thế cho chat_session.send_message, xử lý chuyển đổi history
        '''
        prompt_text = ""

        # 1. Chuyển đổi content input thành text
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

        # 2. QUAN TRỌNG: Chuyển đổi self.history từ format Google (dict) sang LangChain (Message objects)
        # Nếu không có bước này, LLM sẽ không đọc được Prompt hướng dẫn ban đầu.
        messages_to_send = []
        for item in self.history:
            if isinstance(item, dict):
                # Trích xuất text từ cấu trúc dict: {"role": "...", "parts": [{"text": "..."}]}
                text_content = ""
                if "parts" in item:
                    for part in item["parts"]:
                        if isinstance(part, dict):
                            text_content += part.get("text", "")
                        else:
                            text_content += str(part)

                role = item.get("role", "user")
                if role == "user":
                    messages_to_send.append(HumanMessage(content=text_content))
                elif role == "model":
                    messages_to_send.append(AIMessage(content=text_content))
                else:
                    messages_to_send.append(SystemMessage(content=text_content))
            else:
                # Nếu đã là object Message của LangChain thì giữ nguyên
                messages_to_send.append(item)

        # Thêm tin nhắn hiện tại vào cuối danh sách
        messages_to_send.append(HumanMessage(content=prompt_text))

        try:
            # Gửi toàn bộ lịch sử đã chuẩn hóa cho LLM
            ai_message = self.llm.invoke(messages_to_send)
            response_text = ai_message.content
        except Exception as e:
            logging.error(f"LangChain invoke failed: {e}")
            response_text = "NO_AGREEMENT_POSSIBLE: Error calling external LLM"

        # Cập nhật history (giữ nguyên để dùng cho vòng sau)
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



