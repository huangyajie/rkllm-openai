"""
FastAPI router for Chat Completions API.
"""
import json
import uuid
import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat_schema import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
    ChatCompletionChoice, ChatMessage, Usage, ModelList, ModelCard,
    ChatCompletionChunkChoice, DeltaMessage
)
from app.services.chat_service import chat_service
from app.utils.tools import build_prompt_from_messages, parse_tool_calls, parse_thinking
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Chat"])

@router.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models."""
    return ModelList(data=[
        ModelCard(id=settings.MODEL_NAME, created=int(datetime.now().timestamp()))
    ])

@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Handle chat completion requests."""
    if chat_service.is_busy():
        raise HTTPException(status_code=503, detail="Server is busy")

    if not request.messages:
        raise HTTPException(status_code=400, detail="Missing messages")

    # Build prompt
    system_content, user_prompt = build_prompt_from_messages(request.messages)

    # Tools setup
    tools_str = None
    if request.tools:
        tools_str = json.dumps([t for t in request.tools])

    try:
        generator = chat_service.chat(user_prompt, system_content, tools_str, request.enable_thinking)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    if request.stream:
        return StreamingResponse(
            stream_generator(generator, request.model, request.tools is not None),
            media_type="text/event-stream"
        )

    # Collect all output
    full_output_parts = []
    try:
        async for chunk in generator:
            full_output_parts.append(chunk)
    except RuntimeError as e:
        # If busy check failed inside generator initialization (when we start iterating)
        if "busy" in str(e):
            raise HTTPException(status_code=503, detail="Server is busy") from e
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    full_output = "".join(full_output_parts)

    # Parse thinking
    reasoning_content, content_after_thinking = parse_thinking(full_output)

    tool_calls, cleaned_content = parse_tool_calls(
        content_after_thinking) if request.tools else (None, content_after_thinking)

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"

    response = ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(datetime.now().timestamp()),
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(
                    role="assistant",
                    content=cleaned_content,
                    reasoning_content=reasoning_content,
                    tool_calls=tool_calls
                ),
                finish_reason=finish_reason
            )
        ],
        usage=Usage()  # Usage stats not fully tracked in this simple impl
    )
    return response

async def stream_generator(generator, model_name, has_tools):
    """Generate streaming response chunks."""
    buffer = ""
    in_tool_call = False
    in_thinking = False

    tool_start_tag = "<tool_call>"
    tool_end_tag = "</tool_call>"
    think_start_tag = "<think>"
    think_end_tag = "</think>"

    full_output = ""

    try:
        async for char_chunk in generator:
            full_output += char_chunk
            buffer += char_chunk

            # 1. Handle Tool Call State
            if in_tool_call:
                if buffer.endswith(tool_end_tag):
                    in_tool_call = False
                    buffer = ""
                continue

            # 2. Handle Thinking State
            if in_thinking:
                if buffer.endswith(think_end_tag):
                    content = buffer[:-len(think_end_tag)]
                    if content:
                        yield create_chunk(model_name, reasoning_content=content)
                    in_thinking = False
                    buffer = ""
                else:
                    # Output safe part of buffer
                    if len(buffer) > len(think_end_tag):
                        to_yield = buffer[:-len(think_end_tag)]
                        yield create_chunk(model_name, reasoning_content=to_yield)
                        buffer = buffer[-len(think_end_tag):]
                continue

            # 3. Handle Normal Content / Tag Detection

            # Check against enabled tags
            is_tool_prefix = has_tools and (tool_start_tag.startswith(buffer))
            is_think_prefix = think_start_tag.startswith(buffer)

            if is_tool_prefix or is_think_prefix:
                # Exact match check
                if has_tools and buffer == tool_start_tag:
                    in_tool_call = True
                    buffer = ""
                elif buffer == think_start_tag:
                    in_thinking = True
                    buffer = ""
                # If just a prefix, we wait for more chars (continue loop)
                continue

            # Not a tag prefix, output buffer as content
            yield create_chunk(model_name, content=buffer)
            buffer = ""

        # End of stream
        # Flush buffer if any (and not in a special state that consumes it)
        if buffer:
            if in_thinking:
                # If stream ends while thinking, just yield what we have as reasoning
                yield create_chunk(model_name, reasoning_content=buffer)
            elif not in_tool_call:
                yield create_chunk(model_name, content=buffer)

        # Parse full output for tools at the end to send tool_calls chunk
        if has_tools:
            tool_calls, _ = parse_tool_calls(full_output)
            if tool_calls:
                yield create_chunk(model_name, tool_calls=tool_calls)
                yield create_chunk(model_name, finish_reason="tool_calls")
            else:
                yield create_chunk(model_name, finish_reason="stop")
        else:
            yield create_chunk(model_name, finish_reason="stop")

        yield "data: [DONE]\n\n"

    except RuntimeError as e:
        # Catch busy error if it happens during iteration start
        # pylint: disable=f-string-without-interpolation
        error_json = json.dumps({"error": str(e)})
        yield f"data: {error_json}\n\n"
    except Exception as e: # pylint: disable=broad-except
        logger.error("Error in stream: %s", e)
        yield "data: [DONE]\n\n"

def create_chunk(model, content=None, reasoning_content=None, tool_calls=None, finish_reason=None):
    """Helper to create a response chunk."""
    delta = DeltaMessage()
    if content:
        delta.content = content
    if reasoning_content:
        delta.reasoning_content = reasoning_content
    if tool_calls:
        delta.tool_calls = tool_calls

    chunk = ChatCompletionChunk(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(datetime.now().timestamp()),
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=delta,
                finish_reason=finish_reason
            )
        ]
    )
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
