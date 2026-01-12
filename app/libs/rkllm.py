"""
RKLLM wrapper module using ctypes.
"""
import ctypes
import os
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name, too-few-public-methods, bad-indentation
# pylint: disable=too-many-instance-attributes, line-too-long
# pylint: disable=broad-exception-caught

# Load Library
rkllm_lib = None
try:
    lib_path = settings.RKLLM_LIB_PATH
    # Expand user home directory (e.g. ~/libs/...)
    lib_path = os.path.expanduser(lib_path)
    if os.path.exists(lib_path):
        rkllm_lib = ctypes.CDLL(lib_path)
    else:
        # Try finding it relative to CWD if it's just a filename or relative path
        if os.path.exists(os.path.abspath(lib_path)):
            rkllm_lib = ctypes.CDLL(os.path.abspath(lib_path))
        else:
            logger.warning("Warning: RKLLM library not found at %s", lib_path)
except Exception as e: # pylint: disable=broad-except
    logger.error("Error loading RKLLM library: %s", e)

# Definitions
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

class LLMCallState:
    """LLM Call State Constants"""
    RKLLM_RUN_NORMAL = 0
    RKLLM_RUN_WAITING = 1
    RKLLM_RUN_FINISH = 2
    RKLLM_RUN_ERROR = 3

class RKLLMInputType:
    """RKLLM Input Type Constants"""
    RKLLM_INPUT_PROMPT = 0
    RKLLM_INPUT_TOKEN = 1
    RKLLM_INPUT_EMBED = 2
    RKLLM_INPUT_MULTIMODAL = 3

class RKLLMInferMode:
    """RKLLM Infer Mode Constants"""
    RKLLM_INFER_GENERATE = 0
    RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
    RKLLM_INFER_GET_LOGITS = 2

class RKLLMExtendParam(ctypes.Structure):
    """RKLLM Extend Parameters"""
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    """RKLLM Parameters"""
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    """RKLLM LoRA Adapter"""
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    """RKLLM Embedding Input"""
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    """RKLLM Token Input"""
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModalInput(ctypes.Structure):
    """RKLLM MultiModal Input"""
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    """RKLLM Input Union"""
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModalInput)
    ]

class RKLLMInput(ctypes.Structure):
    """RKLLM Input Structure"""
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int), # RKLLMInputType
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    """RKLLM LoRA Parameters"""
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    """RKLLM Prompt Cache Parameters"""
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    """RKLLM Inference Parameters"""
    _fields_ = [
        ("mode", ctypes.c_int), # RKLLMInferMode
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    """RKLLM Result Last Hidden Layer"""
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    """RKLLM Result Logits"""
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMPerfStat(ctypes.Structure):
    """RKLLM Performance Stats"""
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]

class RKLLMResult(ctypes.Structure):
    """RKLLM Result Structure"""
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]

# Callback Type
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

class RKLLM:
    """
    RKLLM Model Wrapper.
    """
    def __init__(self, model_path: str, platform: str = "rk3588",
                 lora_model_path: Optional[str] = None, prompt_cache_path: Optional[str] = None,
                 callback_func=None):
        if rkllm_lib is None:
            raise RuntimeError("RKLLM library not loaded. Check lib/librkllmrt.so")

        self.callback = callback_type(callback_func) if callback_func else None

        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(os.path.expanduser(model_path), "utf-8")
        rkllm_param.max_context_len = settings.MAX_CONTEXT_LEN
        rkllm_param.max_new_tokens = settings.MAX_NEW_TOKENS
        rkllm_param.skip_special_token = True
        rkllm_param.n_keep = -1
        rkllm_param.top_k = settings.TOP_K
        rkllm_param.top_p = settings.TOP_P
        rkllm_param.temperature = settings.TEMPERATURE
        rkllm_param.repeat_penalty = settings.REPEAT_PENALTY
        rkllm_param.frequency_penalty = settings.FREQUENCY_PENALTY
        rkllm_param.presence_penalty = settings.PRESENCE_PENALTY
        rkllm_param.mirostat = 0
        rkllm_param.mirostat_tau = 5.0
        rkllm_param.mirostat_eta = 0.1
        rkllm_param.is_async = False
        rkllm_param.img_start = "".encode("utf-8")
        rkllm_param.img_end = "".encode("utf-8")
        rkllm_param.img_content = "".encode("utf-8")

        rkllm_param.extend_param.base_domain_id = 0
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = 0
        rkllm_param.extend_param.enabled_cpus_num = 4

        if platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)

        self.handle = RKLLM_Handle_t()

        # Init
        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int

        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), self.callback)
        if ret != 0:
            raise RuntimeError("rkllm init failed")

        # Run Bindings
        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int

        # Template Bindings
        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int

        # Tools Bindings
        self.set_function_tools_ = rkllm_lib.rkllm_set_function_tools
        self.set_function_tools_.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_function_tools_.restype = ctypes.c_int

        # Destroy Bindings
        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int

        self.rkllm_abort = rkllm_lib.rkllm_abort

        # LoRA
        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "default"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p(os.path.expanduser(lora_model_path).encode("utf-8"))
            lora_adapter.lora_adapter_name = ctypes.c_char_p(lora_adapter_name.encode("utf-8"))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p(lora_adapter_name.encode("utf-8"))

        # Infer Params
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0

        # Prompt Cache
        if prompt_cache_path:
            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p(os.path.expanduser(prompt_cache_path).encode("utf-8")))

        self.tools = None

    def set_function_tools(self, system_prompt: str, tools: str, tool_response_str: str):
        """Set function calling tools."""
        if self.tools is None or not self.tools == tools:
            self.tools = tools
            self.set_function_tools_(
                self.handle,
                ctypes.c_char_p(system_prompt.encode("utf-8")),
                ctypes.c_char_p(tools.encode("utf-8")),
                ctypes.c_char_p(tool_response_str.encode("utf-8"))
            )

    def run(self, role: str, enable_thinking: bool, prompt: str):
        """Run inference."""
        rkllm_input = RKLLMInput()
        rkllm_input.role = role.encode("utf-8") if role else "user".encode("utf-8")
        rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking)
        rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
        rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode("utf-8"))

        self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)

    def abort(self):
        """Abort inference."""
        return self.rkllm_abort(self.handle)

    def release(self):
        """Release resources."""
        self.rkllm_destroy(self.handle)
