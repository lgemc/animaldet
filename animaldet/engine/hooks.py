"""
Base hooks for training and evaluation callbacks.

Hooks are called at specific points during training to enable
extensible behavior like logging, metric tracking, checkpointing, etc.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class Hook(ABC):
    """
    Base class for all hooks.

    Hooks are called at specific training/evaluation events to enable
    modular extensions like logging, metrics tracking, checkpointing, etc.
    """

    def before_train(self, trainer: Any) -> None:
        """Called once before training starts."""
        pass

    def after_train(self, trainer: Any) -> None:
        """Called once after training completes."""
        pass

    def before_epoch(self, trainer: Any, epoch: int) -> None:
        """Called before each epoch starts."""
        pass

    def after_epoch(self, trainer: Any, epoch: int) -> None:
        """Called after each epoch completes."""
        pass

    def before_step(self, trainer: Any, step: int) -> None:
        """Called before each training step."""
        pass

    def after_step(self, trainer: Any, step: int) -> None:
        """Called after each training step."""
        pass

    def before_eval(self, trainer: Any) -> None:
        """Called before evaluation starts."""
        pass

    def after_eval(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        """Called after evaluation completes."""
        pass


class HookManager:
    """
    Manages and calls multiple hooks in order.

    Example:
        manager = HookManager([ConsoleLogger(), TensorBoardLogger()])
        manager.before_train(trainer)
        manager.after_step(trainer, step=0)
    """

    def __init__(self, hooks: Optional[list[Hook]] = None):
        self.hooks = hooks or []

    def add_hook(self, hook: Hook) -> None:
        """Add a hook to the manager."""
        self.hooks.append(hook)

    def before_train(self, trainer: Any) -> None:
        for hook in self.hooks:
            hook.before_train(trainer)

    def after_train(self, trainer: Any) -> None:
        for hook in self.hooks:
            hook.after_train(trainer)

    def before_epoch(self, trainer: Any, epoch: int) -> None:
        for hook in self.hooks:
            hook.before_epoch(trainer, epoch)

    def after_epoch(self, trainer: Any, epoch: int) -> None:
        for hook in self.hooks:
            hook.after_epoch(trainer, epoch)

    def before_step(self, trainer: Any, step: int) -> None:
        for hook in self.hooks:
            hook.before_step(trainer, step)

    def after_step(self, trainer: Any, step: int) -> None:
        for hook in self.hooks:
            hook.after_step(trainer, step)

    def before_eval(self, trainer: Any) -> None:
        for hook in self.hooks:
            hook.before_eval(trainer)

    def after_eval(self, trainer: Any, metrics: Dict[str, Any]) -> None:
        for hook in self.hooks:
            hook.after_eval(trainer, metrics)
