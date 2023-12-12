from typing import Optional

from pydantic import BaseModel


class User(BaseModel):
    id: str
    properties: dict


class UserRequest(BaseModel):
    utterance: str
    callbackUrl: Optional[str] = None
    user: User


class Intent(BaseModel):
    name: str


class ChatbotRequest(BaseModel):
    userRequest: UserRequest
    intent: Intent
    action: dict
