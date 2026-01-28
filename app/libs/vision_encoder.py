"""
Vision Encoder module for processing images.
"""
import logging
import numpy as np
import cv2
from app.libs.rknn import RKNN

logger = logging.getLogger(__name__)

class VisionEncoder:
    """
    Vision Encoder using RKNN.
    """
    def __init__(self, model_path: str, core_num: int = 3):
        self.model_path = model_path
        self.core_num = core_num
        self.rknn = RKNN(model_path, core_num)
        logger.info("Vision Encoder initialized with model: %s", model_path)

    def _expand2square(self, img: np.ndarray, background_color=(127.5, 127.5, 127.5)) -> np.ndarray:
        """Expand image to square with padding."""
        height, width = img.shape[:2]
        if width == height:
            return img.copy()

        size = max(width, height)
        result = np.full((size, size, 3), background_color, dtype=np.uint8)

        x_offset = (size - width) // 2
        y_offset = (size - height) // 2

        result[y_offset:y_offset + height, x_offset:x_offset + width] = img
        return result

    def preprocess(self, image_data: bytes) -> np.ndarray:
        """Preprocess image from bytes."""
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image")

        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Expand to square
        img = self._expand2square(img)

        # Resize
        img = cv2.resize(img, (self.rknn.model_width, self.rknn.model_height), interpolation=cv2.INTER_LINEAR)

        return img

    def encode(self, image_data: bytes):
        """Encode image and return embeddings and metadata."""
        processed_img = self.preprocess(image_data)
        embeddings = self.rknn.run(processed_img.tobytes())

        return {
            "embeddings": embeddings,
            "n_image_tokens": self.rknn.model_image_token,
            "image_width": self.rknn.model_width,
            "image_height": self.rknn.model_height,
            "n_output": self.rknn.io_num.n_output
        }

    def release(self):
        """Release resources."""
        self.rknn.release()
