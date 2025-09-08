"""
Nonlinearity functions for neural network activation.
"""

import numpy as np
from enum import Enum
from typing import Callable


class NonlinearityType(Enum):
    """Enum for different nonlinearity activation functions."""
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    HEAVISIDE = "heaviside"
    LINEAR = "linear"
    TANH_RELU = "tanh_relu"  # Custom: tanh followed by ReLU (positive part only)


def get_nonlinearity_function(nonlinearity_type: NonlinearityType) -> Callable[[np.ndarray], np.ndarray]:
    """
    Get the activation function corresponding to the nonlinearity type.
    
    Args:
        nonlinearity_type: The type of nonlinearity to use
        
    Returns:
        The activation function
    """
    if nonlinearity_type == NonlinearityType.TANH:
        return np.tanh
    elif nonlinearity_type == NonlinearityType.SIGMOID:
        return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    elif nonlinearity_type == NonlinearityType.RELU:
        return lambda x: np.maximum(0, x)
    elif nonlinearity_type == NonlinearityType.HEAVISIDE:
        return lambda x: (x > 0).astype(float)
    elif nonlinearity_type == NonlinearityType.LINEAR:
        return lambda x: x
    elif nonlinearity_type == NonlinearityType.TANH_RELU:
        return lambda x: np.where(np.tanh(x) > 0, np.tanh(x), 0)
    else:
        raise ValueError(f"Unknown nonlinearity type: {nonlinearity_type}")