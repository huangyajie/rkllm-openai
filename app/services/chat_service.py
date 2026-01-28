"""
Chat service module handling model inference and threading.
"""
import base64
import threading
import queue
import asyncio
import logging
from typing import Optional, AsyncGenerator, Any, Dict
import numpy as np
from app.libs.rkllm import RKLLM, LLMCallState
from app.libs.vision_encoder import VisionEncoder
from app.core.config import settings

logger = logging.getLogger(__name__)

class ChatService:
    """
    Singleton service to manage RKLLM model interactions.
    """
    _instance = None
    _lock = threading.Lock()
    _is_blocking = False

    def __init__(self):
        self.rkllm_model: Optional[RKLLM] = None
        self.vision_model: Optional[VisionEncoder] = None
        self.output_queue = queue.Queue()
        self.current_state = -1
        self.system_prompt = ""
        self.generated_text = []

    @classmethod
    def get_instance(cls):
        """Returns the singleton instance of ChatService."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def initialize_model(self):
        """Initializes the RKLLM model."""
        if self.rkllm_model is not None:
            return

        logger.info("Loading RKLLM model from %s...", settings.MODEL_PATH)
        self.rkllm_model = RKLLM(
            model_path=settings.MODEL_PATH,
            platform=settings.TARGET_PLATFORM,
            lora_model_path=settings.LORA_MODEL_PATH if settings.LORA_MODEL_PATH else None,
            prompt_cache_path=settings.PROMPT_CACHE_PATH if settings.PROMPT_CACHE_PATH else None,
            callback_func=self._callback
        )
        logger.info("RKLLM model loaded.")

        if settings.VISION_MODEL_PATH:
            logger.info("Loading Vision model from %s...", settings.VISION_MODEL_PATH)
            try:
                self.vision_model = VisionEncoder(
                    model_path=settings.VISION_MODEL_PATH,
                    core_num=settings.RKNN_CORE_NUM
                )
                logger.info("Vision model loaded.")
            except Exception as e: # pylint: disable=broad-exception-caught
                logger.error("Failed to load Vision model: %s", e)

    def _callback(self, result, userdata, state):
        """Callback function for RKLLM inference."""
        # pylint: disable=unused-argument
        if state == LLMCallState.RKLLM_RUN_FINISH:
            self.current_state = state
            self.output_queue.put(("finish", None))
        elif state == LLMCallState.RKLLM_RUN_ERROR:
            self.current_state = state
            self.output_queue.put(("error", None))
        elif state == LLMCallState.RKLLM_RUN_NORMAL:
            self.current_state = state
            text = result.contents.text.decode("utf-8")
            self.generated_text.append(text)
            self.output_queue.put(("text", text))
        return 0

    def is_busy(self):
        """Checks if the service is currently processing a request."""
        return self._is_blocking

    def _process_multimodal_input(self, user_prompt: Any) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Processes user input, handling text and images for multimodal requests.
        Returns: (final_prompt_string, multimodal_data_dict)
        """
        if not isinstance(user_prompt, list):
            return user_prompt, None

        text_parts = []
        image_embeddings_list = []
        vision_metadata = {}

        for part in user_prompt:
            if part["type"] == "text":
                text_parts.append(part["text"])
            elif part["type"] == "image_url":
                if self.vision_model is None:
                    raise RuntimeError("Vision model not initialized for multimodal request")

                image_url = part["image_url"]["url"]
                image_data = None
                if image_url.startswith("data:image"):
                    # Base64
                    try:
                        base64_data = image_url.split(",")[1]
                        image_data = base64.b64decode(base64_data)
                    except Exception as e: # pylint: disable=broad-exception-caught
                        logger.error("Failed to decode base64 image: %s", e)
                        continue
                else:
                    logger.warning("External image URLs not yet supported, use base64")
                    continue

                if image_data:
                    logger.info("Processing image %d for multimodal input...", len(image_embeddings_list) + 1)
                    enc_result = self.vision_model.encode(image_data)
                    image_embeddings_list.append(enc_result["embeddings"])

                    if not vision_metadata:
                        vision_metadata = enc_result

                    text_parts.append("<image>")

        final_prompt = "".join(text_parts)
        multimodal_data = None

        if image_embeddings_list:
            final_embeddings = np.concatenate(image_embeddings_list)
            multimodal_data = {
                "embeddings": final_embeddings,
                "n_image": len(image_embeddings_list),
                "n_image_tokens": vision_metadata["n_image_tokens"],
                "image_width": vision_metadata["image_width"],
                "image_height": vision_metadata["image_height"]
            }

        return final_prompt, multimodal_data

    async def chat(self, user_prompt: Any, system_prompt: str = "",
                   tools: str = None, enable_thinking: bool = False) -> AsyncGenerator[str, None]:
        """
        Generates chat response asynchronously.
        """
        if self.rkllm_model is None:
            raise RuntimeError("Model not initialized. Please check logs for startup errors.")

        # Check busy and set blocking atomically
        with self._lock:
            if self._is_blocking:
                raise RuntimeError("Server is busy")
            self._is_blocking = True

        thread = None
        try:
            self.output_queue = queue.Queue()
            self.generated_text = []
            self.current_state = -1
            self.system_prompt = system_prompt

            # Process input (Text + Images)
            final_prompt, multimodal_data = self._process_multimodal_input(user_prompt)

            # Always update function tools state (even if None, to clear previous state)
            self.rkllm_model.set_function_tools(
                system_prompt=system_prompt,
                tools=tools,
                tool_response_str="tool_response"
            )

            # Start inference in a separate thread
            thread = threading.Thread(
                target=self.rkllm_model.run,
                args=("user", enable_thinking, final_prompt, multimodal_data)
            )
            thread.start()

            # Yield results from queue
            while thread.is_alive():
                try:
                    # Non-blocking get with async sleep
                    item = self.output_queue.get_nowait()
                    msg_type, content = item
                    if msg_type == "text":
                        yield content
                    elif msg_type == "finish":
                        break
                    elif msg_type == "error":
                        raise RuntimeError("RKLLM Run Error")
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

            # Check for remaining items
            while not self.output_queue.empty():
                try:
                    item = self.output_queue.get_nowait()
                    msg_type, content = item
                    if msg_type == "text":
                        yield content
                except queue.Empty:
                    break

        except asyncio.CancelledError:
            logger.info("Request cancelled by client")
            raise
        except Exception as e: # pylint: disable=broad-except
            logger.error("Error during chat: %s", e)
            raise
        finally:
            if thread and thread.is_alive():
                logger.warning("Aborting RKLLM inference...")
                self.rkllm_model.abort()

            with self._lock:
                self._is_blocking = False

    def abort(self):
        """Aborts the current inference."""
        if self.rkllm_model:
            self.rkllm_model.abort()

    def release(self):
        """Releases RKLLM resources."""
        if self.rkllm_model:
            self.rkllm_model.release()
        if self.vision_model:
            self.vision_model.release()

chat_service = ChatService.get_instance()
