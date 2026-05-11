from pydantic import BaseModel, field_validator
from typing import Literal


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def must_have_messages(cls, v):
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # Single letter code: A/B/C/D/E/K/P/S


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation] = []
    end_of_conversation: bool = False
