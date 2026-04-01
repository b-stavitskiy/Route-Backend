from typing import Any

from pydantic import BaseModel, Field


class UserBase(BaseModel):
    email: str
    name: str | None = None


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: str | None = None
    name: str | None = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: str | None
    plan_tier: str
    is_active: bool
    email_verified: bool

    class Config:
        from_attributes = True


class ApiKeyBase(BaseModel):
    name: str | None = None


class ApiKeyCreate(ApiKeyBase):
    pass


class ApiKeyResponse(BaseModel):
    id: str
    key_prefix: str
    name: str | None
    plan_tier: str
    created_at: str
    last_used_at: str | None

    class Config:
        from_attributes = True


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float | None = Field(default=None, ge=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1)
    stream: bool = False
    stop: str | list[str] | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str | None


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    provider: str | None
    choices: list[ChatCompletionChoice]
    usage: UsageInfo


class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str
    tier: str


class UsageResponse(BaseModel):
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float
    models: dict[str, Any]


class ErrorResponse(BaseModel):
    error: dict[str, Any]
