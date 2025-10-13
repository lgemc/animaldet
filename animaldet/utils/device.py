"""Device utilities for handling CUDA, MPS, and CPU devices.

This module provides centralized device handling similar to PyTorch Lightning,
ensuring consistent behavior across training and inference.
"""

import torch
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def get_device(device: Union[str, torch.device], verbose: bool = True) -> torch.device:
    """Get the appropriate torch device with automatic fallback.

    Handles device selection with the following priority:
    1. If requested device is available, use it
    2. If requested device is not available, fallback to CPU with warning
    3. Support for 'cuda', 'mps', 'cpu', and device indices like 'cuda:0'

    Args:
        device: Requested device string ('cuda', 'mps', 'cpu', 'cuda:0', etc.) or torch.device
        verbose: Whether to log device selection (default: True)

    Returns:
        torch.device object ready to use

    Examples:
        >>> device = get_device('cuda')  # Uses CUDA if available, else CPU
        >>> device = get_device('mps')   # Uses MPS if available, else CPU
        >>> device = get_device('cuda:1') # Uses CUDA device 1 if available
    """
    # Handle torch.device input
    if isinstance(device, torch.device):
        device = str(device)

    # Parse device string
    device_str = device.lower().strip()

    # Handle device with index (e.g., 'cuda:0')
    if ':' in device_str:
        device_type, device_index = device_str.split(':', 1)
        device_index = int(device_index)
    else:
        device_type = device_str
        device_index = None

    # Determine actual device to use
    actual_device = None

    if device_type == 'cuda':
        if torch.cuda.is_available():
            if device_index is not None:
                # Check if specific GPU index is available
                if device_index < torch.cuda.device_count():
                    actual_device = torch.device(f'cuda:{device_index}')
                else:
                    if verbose:
                        logger.warning(
                            f"CUDA device {device_index} not available "
                            f"(only {torch.cuda.device_count()} devices found), using CPU"
                        )
                    actual_device = torch.device('cpu')
            else:
                actual_device = torch.device('cuda')
        else:
            if verbose:
                logger.warning("CUDA requested but not available, using CPU")
            actual_device = torch.device('cpu')

    elif device_type == 'mps':
        if torch.backends.mps.is_available():
            actual_device = torch.device('mps')
        else:
            if verbose:
                logger.warning("MPS requested but not available, using CPU")
            actual_device = torch.device('cpu')

    elif device_type == 'cpu':
        actual_device = torch.device('cpu')

    else:
        raise ValueError(
            f"Unknown device type: {device_type}. "
            f"Must be one of: 'cuda', 'mps', 'cpu', or 'cuda:N'"
        )

    if verbose:
        logger.info(f"Using device: {actual_device}")

    return actual_device


def move_to_device(
    obj: Union[torch.Tensor, torch.nn.Module, dict, list],
    device: Union[str, torch.device]
) -> Union[torch.Tensor, torch.nn.Module, dict, list]:
    """Move tensor, module, or nested structure to device.

    Handles moving various PyTorch objects to the specified device:
    - Tensors
    - Modules (models)
    - Dictionaries with tensor/module values
    - Lists of tensors/modules
    - Nested combinations of the above

    Args:
        obj: Object to move (tensor, module, dict, list, or nested structure)
        device: Target device

    Returns:
        Object moved to device (same type as input)

    Examples:
        >>> model = move_to_device(model, 'cuda')
        >>> batch = move_to_device({'images': x, 'labels': y}, device)
    """
    device = get_device(device, verbose=False)

    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)

    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    elif isinstance(obj, (list, tuple)):
        moved = [move_to_device(item, device) for item in obj]
        return type(obj)(moved)  # Preserve list/tuple type

    else:
        # Return as-is for non-torch objects
        return obj


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seed for reproducibility across all devices.

    Sets seeds for:
    - Python random
    - NumPy random
    - PyTorch CPU
    - PyTorch CUDA (if available)
    - PyTorch MPS (if available)

    Args:
        seed: Random seed value
        deterministic: If True, use deterministic algorithms (slower but reproducible)

    Example:
        >>> set_seed(42, deterministic=True)
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if torch.backends.mps.is_available():
        # MPS uses the same manual_seed as CPU
        torch.manual_seed(seed)


def supports_autocast(device: Union[str, torch.device]) -> bool:
    """Check if device supports autocast for mixed precision training.

    Args:
        device: Device to check

    Returns:
        True if autocast is supported, False otherwise

    Note:
        - CUDA: Supports autocast with bfloat16/float16
        - CPU: Supports autocast with bfloat16 only
        - MPS: Does not support autocast (as of PyTorch 2.x)
    """
    device = get_device(device, verbose=False)
    device_type = device.type

    if device_type == 'cuda':
        return True
    elif device_type == 'cpu':
        return True  # CPU supports bfloat16 autocast
    elif device_type == 'mps':
        return False  # MPS doesn't support autocast yet
    else:
        return False


def get_autocast_kwargs(
    device: Union[str, torch.device],
    enabled: bool = True,
    dtype: Optional[torch.dtype] = None
) -> dict:
    """Get autocast kwargs for mixed precision training.

    Automatically determines the best autocast settings for the device:
    - CUDA: Uses bfloat16 if available, else float16
    - CPU: Uses bfloat16
    - MPS: Disables autocast (not supported)

    Args:
        device: Target device
        enabled: Whether autocast should be enabled (will be disabled for MPS)
        dtype: Override dtype (if None, uses device-appropriate default)

    Returns:
        Dictionary of kwargs for torch.amp.autocast

    Example:
        >>> kwargs = get_autocast_kwargs('cuda', enabled=True)
        >>> with torch.amp.autocast(**kwargs):
        ...     output = model(input)
    """
    device = get_device(device, verbose=False)
    device_type = device.type

    # MPS doesn't support autocast
    if device_type == 'mps':
        return {'device_type': 'cpu', 'enabled': False}

    # Determine appropriate dtype
    if dtype is None:
        if device_type == 'cuda':
            # Use bfloat16 for CUDA if available
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:  # CPU
            dtype = torch.bfloat16

    # Map device type for autocast (only cuda and cpu are supported)
    autocast_device_type = 'cuda' if device_type == 'cuda' else 'cpu'

    return {
        'device_type': autocast_device_type,
        'enabled': enabled and supports_autocast(device),
        'dtype': dtype
    }


def get_device_info() -> dict:
    """Get information about available devices.

    Returns:
        Dictionary with device availability information

    Example:
        >>> info = get_device_info()
        >>> print(f"CUDA available: {info['cuda_available']}")
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'mps_available': torch.backends.mps.is_available(),
    }

    if info['cuda_available']:
        info['cuda_devices'] = [
            {
                'index': i,
                'name': torch.cuda.get_device_name(i),
                'memory_total': torch.cuda.get_device_properties(i).total_memory,
            }
            for i in range(info['cuda_device_count'])
        ]
        info['cuda_version'] = torch.version.cuda

    return info
