import numpy as np
import pytest
from src.depth.estimator import DepthEstimator

def test_depth_estimator_initialization():
    estimator = DepthEstimator(model_type="MiDaS_small")
    assert estimator.model_type == "MiDaS_small"
    assert estimator._model is None

def test_get_depth_at_point():
    # Create a 10x10 dummy depth map
    depth_map = np.zeros((10, 10))
    depth_map[4:6, 4:6] = 1.0 # center is close (1.0)
    
    # Check center (should average out to > 0 since it contains the 1.0 block)
    depth = DepthEstimator.get_depth_at_point(depth_map, (5, 5), window_size=3)
    assert depth > 0.0
    
    # Check edge cases
    depth = DepthEstimator.get_depth_at_point(depth_map, (0, 0), window_size=3)
    assert depth == 0.0

def test_get_depth_at_point_out_of_bounds():
    depth_map = np.zeros((10, 10))
    # Shouldn't crash
    depth = DepthEstimator.get_depth_at_point(depth_map, (-1, -1))
    assert depth == 0.0
    
    depth = DepthEstimator.get_depth_at_point(depth_map, (20, 20))
    assert depth == 0.0
