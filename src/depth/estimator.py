"""
Depth Estimator

Wrapper for the MiDaS monocular depth estimation model.
Lazy-loads the model via torch.hub to avoid loading overhead when unused.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class DepthEstimator:
    """Estimates relative depth using the MiDaS model."""

    def __init__(self, model_type: str = "MiDaS_small"):
        """Initialize the Depth Estimator.

        Args:
            model_type: The MiDaS model variant (e.g. "MiDaS_small", "DPT_Hybrid").
                        Defaults to "MiDaS_small" for CPU-friendly inference.
        """
        self.model_type = model_type
        self._model = None
        self._transform = None
        self._device = None
        self._torch = None

    def _load_model(self) -> None:
        """Lazy load the PyTorch model and transforms."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch is not installed. Please install 'torch' and 'timm' "
                "to use depth estimation."
            )

        self._torch = torch
        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        logger.info("Loading MiDaS depth model (%s) on %s...", self.model_type, self._device)
        self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
        self._model.to(self._device)
        self._model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if self.model_type == "DPT_Large" or self.model_type == "DPT_Hybrid":
            self._transform = midas_transforms.dpt_transform
        else:
            self._transform = midas_transforms.small_transform
            
        logger.info("MiDaS model loaded successfully.")

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """Estimate the depth map for a single BGR frame.

        Args:
            frame: A BGR numpy array of shape (H, W, 3).

        Returns:
            A normalized 2D numpy array (H, W) where higher values represent
            closer objects (range ~0.0 to 1.0).
        """
        if self._model is None:
            self._load_model()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self._transform(img).to(self._device)

        with self._torch.no_grad():
            prediction = self._model(input_batch)
            
            prediction = self._torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        
        # Normalize between 0 and 1
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max > d_min:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)
            
        return depth_map

    @staticmethod
    def get_depth_at_point(depth_map: np.ndarray, point: tuple[int, int], window_size: int = 5) -> float:
        """Get the average depth value around a specific point.

        Args:
            depth_map: Normalized 2D depth map.
            point: (x, y) coordinates.
            window_size: Size of the neighborhood to average (e.g. 5x5).

        Returns:
            A float representing the average depth around the point (0.0 to 1.0).
        """
        x, y = point
        h, w = depth_map.shape
        
        half_w = window_size // 2
        y1 = max(0, y - half_w)
        y2 = min(h, y + half_w + 1)
        x1 = max(0, x - half_w)
        x2 = min(w, x + half_w + 1)
        
        if y1 >= y2 or x1 >= x2:
            return 0.0
            
        return float(np.mean(depth_map[y1:y2, x1:x2]))
