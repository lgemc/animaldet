"""Base inference engine for model inference.

This module provides a base class for inference that can be extended by
experiment-specific inference implementations.
"""

import torch
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from abc import ABC, abstractmethod

from animaldet.utils import get_device, move_to_device


class BaseInference(ABC):
    """
    Base inference class for running model inference.

    This class provides a common interface for inference across different
    model architectures and frameworks. Experiment-specific implementations
    should inherit from this class and implement the abstract methods.

    Args:
        model: The model to use for inference
        device: Device to run inference on ('cuda', 'mps', or 'cpu')
        checkpoint_path: Optional path to model checkpoint to load
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        checkpoint_path: Optional[Union[str, Path]] = None
    ):
        self.model = model
        self.device = get_device(device)

        # Move model to device
        self.model = move_to_device(self.model, self.device)

        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

        # Set model to eval mode
        self.model.eval()

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict)

    @torch.no_grad()
    def predict(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], Any],
        **kwargs
    ) -> Any:
        """
        Run inference on input data.

        Args:
            inputs: Input data for inference
            **kwargs: Additional arguments for inference

        Returns:
            Model predictions
        """
        self.model.eval()

        # Move inputs to device
        inputs = move_to_device(inputs, self.device)

        # Run model-specific inference
        return self._predict_impl(inputs, **kwargs)

    @abstractmethod
    def _predict_impl(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], Any],
        **kwargs
    ) -> Any:
        """
        Implementation of prediction logic.

        This method must be implemented by subclasses to define
        experiment-specific inference behavior.

        Args:
            inputs: Input data (already moved to device)
            **kwargs: Additional arguments

        Returns:
            Model predictions
        """
        pass

    @abstractmethod
    def postprocess(self, outputs: Any, **kwargs) -> Any:
        """
        Postprocess model outputs.

        This method must be implemented by subclasses to define
        experiment-specific postprocessing.

        Args:
            outputs: Raw model outputs
            **kwargs: Additional arguments

        Returns:
            Postprocessed predictions
        """
        pass

    def __call__(
        self,
        inputs: Union[torch.Tensor, List[torch.Tensor], Any],
        postprocess: bool = True,
        **kwargs
    ) -> Any:
        """
        Run inference with optional postprocessing.

        Args:
            inputs: Input data
            postprocess: Whether to apply postprocessing
            **kwargs: Additional arguments

        Returns:
            Model predictions (postprocessed if requested)
        """
        outputs = self.predict(inputs, **kwargs)

        if postprocess:
            outputs = self.postprocess(outputs, **kwargs)

        return outputs

    @property
    def state_dict(self) -> Dict[str, Any]:
        """Get model state dictionary."""
        return self.model.state_dict()
