"""
RKNN wrapper module using ctypes.
"""
import ctypes
import os
import logging
import numpy as np
from app.core.config import settings

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name, too-few-public-methods, bad-indentation
# pylint: disable=too-many-instance-attributes, line-too-long
# pylint: disable=broad-exception-caught

# Load Library
rknn_lib = None
try:
    lib_path = settings.RKNN_LIB_PATH
    lib_path = os.path.expanduser(lib_path)
    if os.path.exists(lib_path):
        rknn_lib = ctypes.CDLL(lib_path)
    else:
        if os.path.exists(os.path.abspath(lib_path)):
            rknn_lib = ctypes.CDLL(os.path.abspath(lib_path))
        else:
            logger.warning("Warning: RKNN library not found at %s", lib_path)
except Exception as e:
    logger.error("Error loading RKNN library: %s", e)

# Definitions
# rknn_context is uint32_t on arm (32bit), uint64_t on aarch64/x86_64 (64bit)
# Use pointer size to determine
if ctypes.sizeof(ctypes.c_void_p) == 8:
    rknn_context = ctypes.c_uint64
else:
    rknn_context = ctypes.c_uint32

RKNN_SUCC = 0
RKNN_MAX_DIMS = 16
RKNN_MAX_NAME_LEN = 256

class RKNNQueryCmd:
    """RKNN Query Commands"""
    RKNN_QUERY_IN_OUT_NUM = 0
    RKNN_QUERY_INPUT_ATTR = 1
    RKNN_QUERY_OUTPUT_ATTR = 2
    RKNN_QUERY_SDK_VERSION = 5

class RKNNTensorType:
    """RKNN Tensor Types"""
    RKNN_TENSOR_FLOAT32 = 0
    RKNN_TENSOR_FLOAT16 = 1
    RKNN_TENSOR_INT8 = 2
    RKNN_TENSOR_UINT8 = 3

class RKNNTensorFormat:
    """RKNN Tensor Formats"""
    RKNN_TENSOR_NCHW = 0
    RKNN_TENSOR_NHWC = 1

class RKNNCoreMask:
    """RKNN Core Mask"""
    RKNN_NPU_CORE_AUTO = 0
    RKNN_NPU_CORE_0 = 1
    RKNN_NPU_CORE_1 = 2
    RKNN_NPU_CORE_2 = 4
    RKNN_NPU_CORE_0_1 = 3
    RKNN_NPU_CORE_0_1_2 = 7

class RKNNInputOutputNum(ctypes.Structure):
    """RKNN Input Output Number"""
    _fields_ = [
        ("n_input", ctypes.c_uint32),
        ("n_output", ctypes.c_uint32)
    ]

class RKNNTensorAttr(ctypes.Structure):
    """RKNN Tensor Attributes"""
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * RKNN_MAX_DIMS),
        ("name", ctypes.c_char * RKNN_MAX_NAME_LEN),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int), # RKNNTensorFormat
        ("type", ctypes.c_int), # RKNNTensorType
        ("qnt_type", ctypes.c_int),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32)
    ]

class RKNNInput(ctypes.Structure):
    """RKNN Input"""
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int),
        ("fmt", ctypes.c_int)
    ]

class RKNNOutput(ctypes.Structure):
    """RKNN Output"""
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32)
    ]

if rknn_lib:
    rknn_lib.rknn_init.argtypes = [ctypes.POINTER(rknn_context), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    rknn_lib.rknn_init.restype = ctypes.c_int

    rknn_lib.rknn_destroy.argtypes = [rknn_context]
    rknn_lib.rknn_destroy.restype = ctypes.c_int

    rknn_lib.rknn_query.argtypes = [rknn_context, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    rknn_lib.rknn_query.restype = ctypes.c_int

    rknn_lib.rknn_inputs_set.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(RKNNInput)]
    rknn_lib.rknn_inputs_set.restype = ctypes.c_int

    rknn_lib.rknn_run.argtypes = [rknn_context, ctypes.c_void_p]
    rknn_lib.rknn_run.restype = ctypes.c_int

    rknn_lib.rknn_outputs_get.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(RKNNOutput), ctypes.c_void_p]
    rknn_lib.rknn_outputs_get.restype = ctypes.c_int

    rknn_lib.rknn_outputs_release.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(RKNNOutput)]
    rknn_lib.rknn_outputs_release.restype = ctypes.c_int

    rknn_lib.rknn_set_core_mask.argtypes = [rknn_context, ctypes.c_int]
    rknn_lib.rknn_set_core_mask.restype = ctypes.c_int

class RKNN:
    """RKNN Model Class"""
    def __init__(self, model_path: str, core_num: int = 0):
        if rknn_lib is None:
            raise RuntimeError("RKNN library not loaded")

        self.ctx = rknn_context(0)
        ret = rknn_lib.rknn_init(ctypes.byref(self.ctx), model_path.encode("utf-8"), 0, 0, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_init failed with ret={ret}")

        # Set Core Mask
        core_mask = RKNNCoreMask.RKNN_NPU_CORE_AUTO
        if core_num == 1:
            core_mask = RKNNCoreMask.RKNN_NPU_CORE_0
        elif core_num == 2:
            core_mask = RKNNCoreMask.RKNN_NPU_CORE_0_1
        elif core_num == 3:
            core_mask = RKNNCoreMask.RKNN_NPU_CORE_0_1_2

        ret = rknn_lib.rknn_set_core_mask(self.ctx, core_mask)
        if ret != RKNN_SUCC:
            logger.warning("Failed to set NPU core mask to %d (ret=%d). Falling back to AUTO mode. "
                           "Please check if RKNN_CORE_NUM=%d is supported by your hardware (e.g. RK3576 only has 2 cores).", 
                           core_mask, ret, core_num)
            rknn_lib.rknn_set_core_mask(self.ctx, RKNNCoreMask.RKNN_NPU_CORE_AUTO)

        # Query IO num
        self.io_num = RKNNInputOutputNum()
        rknn_lib.rknn_query(self.ctx, RKNNQueryCmd.RKNN_QUERY_IN_OUT_NUM, ctypes.byref(self.io_num), ctypes.sizeof(self.io_num))

        # Query attributes
        self.input_attrs = []
        for i in range(self.io_num.n_input):
            attr = RKNNTensorAttr()
            attr.index = i
            rknn_lib.rknn_query(self.ctx, RKNNQueryCmd.RKNN_QUERY_INPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
            self.input_attrs.append(attr)

        self.output_attrs = []
        for i in range(self.io_num.n_output):
            attr = RKNNTensorAttr()
            attr.index = i
            rknn_lib.rknn_query(self.ctx, RKNNQueryCmd.RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(attr), ctypes.sizeof(attr))
            self.output_attrs.append(attr)

        # Get model dimensions
        if self.input_attrs[0].fmt == RKNNTensorFormat.RKNN_TENSOR_NCHW:
            self.model_channel = self.input_attrs[0].dims[1]
            self.model_height = self.input_attrs[0].dims[2]
            self.model_width = self.input_attrs[0].dims[3]
        else:
            self.model_height = self.input_attrs[0].dims[1]
            self.model_width = self.input_attrs[0].dims[2]
            self.model_channel = self.input_attrs[0].dims[3]

        # Get embed info
        self.model_image_token = 0
        self.model_embed_size = 0
        for i in range(4):
            if self.output_attrs[0].dims[i] > 1:
                self.model_image_token = self.output_attrs[0].dims[i]
                self.model_embed_size = self.output_attrs[0].dims[i+1]
                break

    def run(self, img_data):
        """Run RKNN inference"""
        # img_data should be bytes or a pointer
        inputs = (RKNNInput * 1)()
        inputs[0].index = 0
        inputs[0].type = RKNNTensorType.RKNN_TENSOR_UINT8
        inputs[0].fmt = RKNNTensorFormat.RKNN_TENSOR_NHWC
        inputs[0].size = self.model_width * self.model_height * self.model_channel
        inputs[0].buf = ctypes.cast(img_data, ctypes.c_void_p)

        ret = rknn_lib.rknn_inputs_set(self.ctx, 1, inputs)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_inputs_set failed with ret={ret}")

        ret = rknn_lib.rknn_run(self.ctx, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_run failed with ret={ret}")

        outputs = (RKNNOutput * self.io_num.n_output)()
        for i in range(self.io_num.n_output):
            outputs[i].index = i
            outputs[i].want_float = 1
            outputs[i].is_prealloc = 0

        ret = rknn_lib.rknn_outputs_get(self.ctx, self.io_num.n_output, outputs, None)
        if ret != RKNN_SUCC:
            raise RuntimeError(f"rknn_outputs_get failed with ret={ret}")

        # Post process
        res = []
        if self.io_num.n_output == 1:
            # Copy output
            data = ctypes.string_at(outputs[0].buf, outputs[0].size)
            res = np.frombuffer(data, dtype=np.float32).copy()
        else:
            # Concat outputs as in official demo
            # Official demo logic:
            # for (int i = 0; i < app_ctx->model_image_token; i++) {
            #     for (int j = 0; j < app_ctx->io_num.n_output; j++) {
            #         memcpy(out_result + i * app_ctx->io_num.n_output * app_ctx->model_embed_size + j * app_ctx->model_embed_size,
            #                (float*)(outputs[j].buf) + i * app_ctx->model_embed_size, sizeof(float) * app_ctx->model_embed_size);
            #     }
            # }
            # In numpy this is more like stacking and reshaping
            output_arrays = []
            for i in range(self.io_num.n_output):
                data = ctypes.string_at(outputs[i].buf, outputs[i].size)
                output_arrays.append(np.frombuffer(data, dtype=np.float32).reshape(self.model_image_token, self.model_embed_size))

            # Stack and transpose/reshape to match the interleaved format
            stacked = np.stack(output_arrays, axis=1) # (tokens, n_outputs, embed_size)
            res = stacked.flatten().copy()

        rknn_lib.rknn_outputs_release(self.ctx, self.io_num.n_output, outputs)
        return res

    def release(self):
        """Release RKNN context"""
        if self.ctx:
            rknn_lib.rknn_destroy(self.ctx)
            self.ctx = None
