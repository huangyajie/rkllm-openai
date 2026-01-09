"""
Chat service module handling model inference and threading.
"""
import threading
import queue
import asyncio
from typing import Optional, AsyncGenerator
from app.libs.rkllm import RKLLM, LLMCallState
from app.core.config import settings

class ChatService:
    """
    Singleton service to manage RKLLM model interactions.
    """
    _instance = None
    _lock = threading.Lock()
    _is_blocking = False

    def __init__(self):
        self.rkllm_model: Optional[RKLLM] = None
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

        print(f"Loading RKLLM model from {settings.MODEL_PATH}...")
        self.rkllm_model = RKLLM(
            model_path=settings.MODEL_PATH,
            platform=settings.TARGET_PLATFORM,
            lora_model_path=settings.LORA_MODEL_PATH if settings.LORA_MODEL_PATH else None,
            prompt_cache_path=settings.PROMPT_CACHE_PATH if settings.PROMPT_CACHE_PATH else None,
            callback_func=self._callback
        )
        print("RKLLM model loaded.")

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

    async def chat(self, user_prompt: str, system_prompt: str = "",
                   tools: str = None) -> AsyncGenerator[str, None]:
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

            if tools:
                self.rkllm_model.set_function_tools(
                    system_prompt=system_prompt,
                    tools=tools,
                    tool_response_str="tool_response"
                )

            # Start inference in a separate thread
            thread = threading.Thread(
                target=self.rkllm_model.run,
                args=("user", False, user_prompt)
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
            print("Request cancelled by client")
            raise
        except Exception as e:
            print(f"Error during chat: {e}")
            raise
        finally:
            if thread and thread.is_alive():
                print("Aborting RKLLM inference...")
                self.rkllm_model.abort()
                # Optionally wait for thread to cleanup?
                # Doing thread.join() here would block the event loop.
                # We assume abort() is sufficient to stop the C++ side eventually.

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

chat_service = ChatService.get_instance()
