"""Registry system for animaldet components.

This module provides a registry pattern similar to HerdNet and Detectron2
for registering and retrieving trainers, models, datasets, and other components.
"""

import sys
from typing import Callable, Optional, Any


class Registry:
    """Registry to map strings to classes.

    This allows components to be registered with a decorator and then
    instantiated from configuration files using their string names.

    Args:
        name: Registry name (e.g., "trainers", "models")
        module_key: Optional module key for updating its __all__ variable

    Example:
        >>> TRAINERS = Registry("trainers")
        >>> @TRAINERS.register()
        ... class MyTrainer:
        ...     pass
        >>> trainer_cls = TRAINERS["MyTrainer"]
    """

    def __init__(self, name: str, module_key: Optional[str] = None) -> None:
        self.name = name
        self.module_key = module_key
        self._registered_objects = {}

    def register(self, name: Optional[str] = None) -> Callable:
        """Register a class with an optional custom name.

        Args:
            name: Optional custom name for registration. If None, uses class name.

        Returns:
            Decorator function that registers the class
        """

        def _register(cls):
            reg_name = name if name is not None else cls.__name__
            if reg_name in self._registered_objects:
                raise ValueError(
                    f"Object '{reg_name}' already registered in {self.name} registry"
                )
            self._registered_objects[reg_name] = cls
            if self.module_key is not None:
                if not hasattr(sys.modules[self.module_key], "__all__"):
                    sys.modules[self.module_key].__all__ = []
                sys.modules[self.module_key].__all__.append(reg_name)
            return cls

        return _register

    def get(self, key: str) -> Any:
        """Get a registered object by name.

        Args:
            key: Name of the registered object

        Returns:
            The registered object

        Raises:
            KeyError: If the key is not found in the registry
        """
        if key not in self._registered_objects:
            raise KeyError(
                f"'{key}' not found in {self.name} registry. "
                f"Available: {list(self._registered_objects.keys())}"
            )
        return self._registered_objects[key]

    @property
    def registry_names(self) -> list:
        """Get list of all registered names."""
        return list(self._registered_objects.keys())

    def __getitem__(self, key: str) -> Any:
        """Get a registered object using bracket notation."""
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if a key is registered."""
        return key in self._registered_objects

    def __len__(self) -> int:
        """Get number of registered objects."""
        return len(self._registered_objects)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name='{self.name}', "
            f"objects={list(self._registered_objects.keys())})"
        )
