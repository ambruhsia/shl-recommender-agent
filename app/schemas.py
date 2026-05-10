from __future__ import annotations

from typing import List
from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    role: str       # "user" | "assistant" | "system"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_turn_count(self) -> ChatRequest:
        user_turns = sum(1 for m in self.messages if m.role == "user")
        if user_turns > 8:
            raise ValueError("Conversation exceeds maximum of 8 user turns")
        return self


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @model_validator(mode="after")
    def clamp_recommendations(self) -> ChatResponse:
        if len(self.recommendations) > 10:
            self.recommendations = self.recommendations[:10]
        return self
