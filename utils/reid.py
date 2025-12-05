"""Lightweight ReID extractor.

Provides ReIDExtractor.extract(frame_bgr, bbox) -> np.ndarray embedding.
This implementation prefers torch if available and falls back to a
fast color-moment based descriptor to avoid hard dependency on PyTorch
for quick local testing.
"""
from typing import Tuple
import numpy as np

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


class ReIDExtractor:
    """Extracts a small appearance descriptor for a bbox in a BGR frame.

    If PyTorch is available a simple pretrained ResNet trunk can be used
    (not provided here) — otherwise returns a normalized color-statistics
    vector (mean, std) per channel.
    """

    def __init__(self, model_path: str = None, embedding_dim: int = 128):
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        # If torch is available we would load weights; for now we don't
        self.use_torch = TORCH_AVAILABLE

    def extract(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Return L2-normalized embedding for the bbox region.

        bbox: [x1,y1,x2,y2]
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))
        if x2 <= x1 or y2 <= y1:
            # empty bbox -> return zero vector
            return np.zeros((self.embedding_dim,), dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]

        if self.use_torch:
            # Placeholder: if a torch model is wired later, use it here.
            # For now we fallback to color stats even if torch exists.
            pass

        # simple color-statistics descriptor: mean/std per channel, plus area
        means = crop.reshape(-1, 3).mean(axis=0)
        stds = crop.reshape(-1, 3).std(axis=0)
        area = np.array([crop.shape[0] * crop.shape[1] / (h * w + 1e-6)])
        desc = np.concatenate([means, stds, area])
        # expand/replicate to requested dim
        vec = np.tile(desc.astype(np.float32), int(np.ceil(self.embedding_dim / desc.size)))[: self.embedding_dim]
        # normalize
        norm = np.linalg.norm(vec) + 1e-6
        return (vec / norm).astype(np.float32)


if __name__ == "__main__":
    # quick smoke test
    import cv2
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    extractor = ReIDExtractor()
    e = extractor.extract(img, (10, 10, 50, 80))
    print(e.shape)
