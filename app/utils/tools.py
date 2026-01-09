"""
Utility functions for tool parsing and prompt building.
"""
import json
import uuid  # pylint: disable=unused-import
import re
from typing import List, Dict, Optional, Tuple, Any
from app.schemas.chat_schema import ChatMessage

def build_prompt_from_messages(messages: List[ChatMessage]) -> Tuple[str, str]:
    """
    Returns (system_content, user_prompt)
    """
    prompt_parts = []
    system_content = ""

    for msg in messages:
        role = msg.role
        content = msg.content if msg.content is not None else ""

        if role == "system":
            system_content = content
        elif role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            # If content is empty but tool_calls exist, we might want to skip or format differently
            # For now, just append content if it exists
            if content:
                prompt_parts.append(f"Assistant: {content}")
            # Note: We are not currently formatting tool_calls into the prompt string
            # relying on the model's internal handling or previous turn context
        elif role == "tool":
            tool_name = msg.name if msg.name else "tool"
            prompt_parts.append(f"Tool {tool_name}: {content}")

    user_prompt = "\n".join(prompt_parts)
    if user_prompt and not user_prompt.endswith("Assistant:"):
        user_prompt += "\nAssistant:"

    return system_content, user_prompt

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
        json_match = re.search(r'\{[^{}]*"name"\s*:\s*"[^"]*"[^{}]*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}', content)
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
