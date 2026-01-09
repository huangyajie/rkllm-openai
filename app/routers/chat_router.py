"""
FastAPI router for Chat Completions API.
"""
import json
import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.chat_schema import (
    ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChunk,
    ChatCompletionChoice, ChatMessage, Usage, ModelList, ModelCard,
    ChatCompletionChunkChoice, DeltaMessage
)
from app.services.chat_service import chat_service
from app.utils.tools import build_prompt_from_messages, parse_tool_calls
from app.core.config import settings

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
        generator = chat_service.chat(user_prompt, system_content, tools_str)
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
    tool_calls, cleaned_content = parse_tool_calls(
        full_output) if request.tools else (None, full_output)

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
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"
    full_output = ""

    try:
        async for char_chunk in generator:
            full_output += char_chunk

            # If not using tools, just stream char by char (or chunk by chunk)
            if not has_tools:
                yield create_chunk(model_name, content=char_chunk)
                continue

            # Tool parsing logic for stream
            buffer += char_chunk

            if in_tool_call:
                if buffer.endswith(end_tag):
                    in_tool_call = False
                    buffer = ""
                continue

            if start_tag.startswith(buffer) or buffer.startswith(start_tag[:len(buffer)]):
                if buffer == start_tag:
                    in_tool_call = True
                continue

            if not start_tag.startswith(buffer):
                yield create_chunk(model_name, content=buffer)
                buffer = ""

        # End of stream
        if buffer and not in_tool_call:
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
    except Exception as e: # pylint: disable=broad-exception-caught
        print(f"Error in stream: {e}")
        yield "data: [DONE]\n\n"

def create_chunk(model, content=None, tool_calls=None, finish_reason=None):
    """Helper to create a response chunk."""
    delta = DeltaMessage()
    if content:
        delta.content = content
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
