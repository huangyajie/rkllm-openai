"""
Pydantic schemas for the Chat API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel

class ChatMessage(BaseModel):
    """Schema for a single chat message."""
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionRequest(BaseModel):
    """Schema for a chat completion request."""
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None

class ChatCompletionChoice(BaseModel):
    """Schema for a single choice in the response."""
    index: int
    message: ChatMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = "stop"

class Usage(BaseModel):
    """Schema for token usage statistics."""
    prompt_tokens: Optional[int] = 0
    completion_tokens: Optional[int] = 0
    total_tokens: Optional[int] = 0

class ChatCompletionResponse(BaseModel):
    """Schema for the full chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None

class DeltaMessage(BaseModel):
    """Schema for a streaming delta message."""
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

class ChatCompletionChunkChoice(BaseModel):
    """Schema for a single choice in a streaming chunk."""
    index: int
    delta: DeltaMessage
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    """Schema for a streaming response chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]

class ModelCard(BaseModel):
    """Schema for model information."""
    id: str
    object: str = "model"
    created: int
    owned_by: str = "rkllm"

class ModelList(BaseModel):
    """Schema for the list of available models."""
    object: str = "list"
    data: List[ModelCard]
