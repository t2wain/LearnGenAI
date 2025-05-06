from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.runnables import ConfigurableField
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel, 
    FakeMessagesListChatModel
)
import json

def get_google_chat(model_name:str = "gemini-2.0-flash") -> BaseChatModel:
    return ChatGoogleGenerativeAI(
        model=model_name
    ).configurable_fields(
        max_output_tokens=ConfigurableField(
            id="max_output_tokens",
            name="LLM Maximum output tokens",
            description="Maximum number of tokens",
        ),
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )


def get_ollama_chat(model_name:str = "gemma3:4b") -> BaseChatModel:
    return ChatOllama(
        model=model_name
    ).configurable_fields(
        temperature=ConfigurableField(
            id="temperature",
            name="LLM Temperature",
            description="The temperature of the LLM",
        )
    )


def get_fake_chat_message(responses: list[BaseMessage]) -> BaseChatModel:
    return FakeMessagesListChatModel(responses=responses).configurable_fields(
        responses=ConfigurableField(
            id="responses",
            name="List of responses to cycle through in order.",
            description="List of responses to cycle through in order.",
        ))


def get_fake_chat_str(responses: list[str]) -> BaseChatModel:
    return FakeListChatModel(responses=responses).configurable_fields(
        responses=ConfigurableField(
            id="responses",
            name="List of responses to cycle through in order.",
            description="List of responses to cycle through in order.",
        ))

def print_dict(val: dict):
    print(json.dumps(val, indent=4))