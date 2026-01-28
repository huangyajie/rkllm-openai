"""
Utility functions for tool parsing and prompt building.
"""
import logging
import json
import re
import uuid
from typing import List, Tuple, Dict, Any, Optional
from app.schemas.chat_schema import ChatMessage

logger = logging.getLogger(__name__)





def build_prompt_from_messages(messages: List[ChatMessage]) -> Tuple[str, Any]:
    """
    Returns (system_content, last_user_content)
    Note: For multimodal, we need the last message's raw content (could be list).
    """
    system_content = ""
    history_parts = []

    # Extract system prompt
    for msg in messages:
        if msg.role == "system":
            system_content = msg.content if isinstance(msg.content, str) else ""
            break

    # We want to pass the last message to chat_service.chat
    # And potentially include history in the prompt if keep_history is 0
    # For now, let's follow a simple pattern:
    # Concatenate all previous messages into a string,
    # and the last message is the "user_prompt".

    if not messages:
        return system_content, ""

    last_msg = messages[-1]

    # Build history from messages before the last one
    for msg in messages[:-1]:
        role = msg.role
        content = msg.content
        if isinstance(content, list):
            # Extract text from multimodal parts for history string
            text_parts = [part["text"] for part in content if part["type"] == "text"]
            content_str = " ".join(text_parts)
        else:
            content_str = content if content else ""

        if role == "user":
            history_parts.append(f"User: {content_str}")
        elif role == "assistant":
            if content_str:
                history_parts.append(f"Assistant: {content_str}")
            if msg.tool_calls:
                for tool_call in msg.tool_calls:
                    func = tool_call.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "{}")
                    # Reconstruct the tool call signature for the model's context
                    tool_call_str = f"<tool_call>{{\"name\": \"{name}\", \"arguments\": {args}}}</tool_call>"
                    history_parts.append(f"Assistant: {tool_call_str}")
        elif role == "tool":
            history_parts.append(f"Tool Observation: {content_str}")

    history_str = "\n".join(history_parts)

    last_content = last_msg.content
    last_role = last_msg.role

    # Format the last message appropriately
    prefix = ""
    if last_role == "user":
        prefix = "User: "
    elif last_role == "tool":
        prefix = "Tool Observation: "

    if history_str:
        if isinstance(last_content, str):
            last_content = f"{history_str}\n{prefix}{last_content}"
        elif isinstance(last_content, list):
            # Prepend history to the first text part of the last message
            for part in last_content:
                if part["type"] == "text":
                    content_text = part["text"]
                    part["text"] = f"{history_str}\n{prefix}{content_text}"
                    break
            else:
                # If no text part found, add one
                last_content.insert(0, {"type": "text", "text": f"{history_str}\n{prefix}"})
    else:
        # No history, just prefix the current message
        if isinstance(last_content, str):
            last_content = f"{prefix}{last_content}"
        # For list content (multimodal), assume the model handles the start
        elif isinstance(last_content, list):
            for part in last_content:
                if part["type"] == "text":
                    part["text"] = f"{prefix}{part["text"]}"
                    break

    return system_content, last_content

def parse_thinking(content: str) -> Tuple[Optional[str], str]:
    """
    Parses <think> content.
    Returns (reasoning_content, cleaned_content)
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        reasoning = match.group(1)
        cleaned = re.sub(pattern, "", content, flags=re.DOTALL)
        return reasoning, cleaned
    return None, content

def parse_tool_calls(content: str) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """
    Returns (tool_calls, cleaned_content)
    """

    empty_dict = {}

    # Try parsing <tool_call> format
    xml_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    matches = re.findall(xml_pattern, content, re.DOTALL)
    if matches:
        try:
            tool_data = json.loads(matches[0])
            # pylint: disable=f-string-without-interpolation, unhashable-member
            tool_calls = [{
                "index": 0,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name", ""),
                    "arguments": json.dumps(tool_data.get("arguments", empty_dict))
                }
            }]
            cleaned_content = re.sub(xml_pattern, "", content, flags=re.DOTALL).strip()
            return tool_calls, cleaned_content
        except json.JSONDecodeError:
            pass

    # Try parsing ```toolcall format
    pattern = r"```toolcall\s*(\{.*?\})\s*```"
    matches = re.findall(pattern, content, re.DOTALL)
    if matches:
        try:
            tool_data = json.loads(matches[0])
            # pylint: disable=f-string-without-interpolation, unhashable-member
            tool_calls = [{
                "index": 0,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name", ""),
                    "arguments": json.dumps(tool_data.get("arguments", empty_dict))
                }
            }]
            cleaned_content = re.sub(pattern, "", content, flags=re.DOTALL).strip()
            return tool_calls, cleaned_content
        except json.JSONDecodeError:
            pass

    # Try JSON detection
    try:
        stripped = content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            tool_data = json.loads(stripped)
            if isinstance(tool_data, dict) and "name" in tool_data and "arguments" in tool_data:
                # pylint: disable=f-string-without-interpolation, unhashable-member
                tool_calls = [{
                    "index": 0,
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": tool_data["name"],
                        "arguments": json.dumps(tool_data["arguments"])
                    }
                }]
                return tool_calls, ""

        # Regex search for JSON inside text
        # pylint: disable=line-too-long
        json_match = re.search(r"\{[^{}]*\"name\"\s*:\s*\"[^\"]*\"[^{}]*\"arguments\"\s*:\s*\{[^{}]*\}[^{}]*\}", content)
        if json_match:
            tool_data = json.loads(json_match.group(0))
            # pylint: disable=f-string-without-interpolation, unhashable-member
            tool_calls = [{
                "index": 0,
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_data["name"],
                    "arguments": json.dumps(tool_data["arguments"])
                }
            }]
            cleaned_content = content[:json_match.start()] + content[json_match.end():]
            return tool_calls, cleaned_content.strip()

    except (json.JSONDecodeError, ValueError):
        pass

    return None, content
