"""Hook builder utilities for loading hooks from configuration.

This module provides utilities to automatically build and register hooks
(e.g., WandB, TensorBoard loggers) from integration configuration files.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf

from animaldet.engine.hooks import Hook
from animaldet.engine.registry import HOOKS


def build_hooks_from_config(
    cfg: Dict[str, Any],
    integrations_dir: Optional[Path] = None
) -> List[Hook]:
    """Build hooks from integration and metric configurations.

    This function automatically loads integration configs and builds the
    corresponding hooks if they are enabled. It also builds metric hooks
    from the evaluator.metrics configuration.

    Args:
        cfg: Main configuration dictionary (should contain integrations config or path)
        integrations_dir: Directory containing integration config files
                         (defaults to configs/integrations/)

    Returns:
        List of initialized Hook instances

    Example:
        >>> cfg = OmegaConf.load("configs/experiments/rfdetr.yaml")
        >>> hooks = build_hooks_from_config(cfg)
        >>> # Returns [TensorBoardLogger(...), F1MetricHook(...)] if enabled
    """
    hooks = []

    # Default integrations directory
    if integrations_dir is None:
        integrations_dir = Path(__file__).parent.parent.parent / "configs" / "integrations"

    # 1. Build integration hooks (TensorBoard, W&B, etc.)
    integrations_cfg = cfg.get("integrations", {})

    # If integrations config is empty, try loading from files
    if not integrations_cfg:
        # Try to load all integration configs
        if integrations_dir.exists():
            for config_file in integrations_dir.glob("*.yaml"):
                hook_name = config_file.stem  # e.g., "wandb", "tensorboard"
                try:
                    hook_cfg = OmegaConf.load(config_file)
                    hook_cfg = OmegaConf.to_container(hook_cfg, resolve=True)
                    integrations_cfg[hook_name] = hook_cfg
                except Exception as e:
                    print(f"Warning: Could not load integration config {config_file}: {e}")

    # Build hooks from integration configs
    for hook_name, hook_cfg in integrations_cfg.items():
        if isinstance(hook_cfg, dict) and hook_cfg.get("enabled", False):
            # Check if hook is registered
            if hook_name in HOOKS:
                try:
                    # Get hook class from registry
                    hook_class = HOOKS[hook_name]

                    # Pass full config to constructor (including 'enabled')
                    init_kwargs = hook_cfg.copy()

                    # Instantiate hook
                    hook = hook_class(**init_kwargs)
                    hooks.append(hook)

                    print(f"Loaded integration hook: {hook_name} ({hook_class.__name__})")
                except Exception as e:
                    print(f"Warning: Failed to initialize hook '{hook_name}': {e}")
            else:
                print(
                    f"Warning: Hook '{hook_name}' not found in registry. "
                    f"Available: {HOOKS.registry_names}"
                )

    # 2. Build metric hooks from evaluator.metrics config
    evaluator_cfg = cfg.get("evaluator", {})
    metrics_cfg = evaluator_cfg.get("metrics", {})

    for metric_name, metric_cfg in metrics_cfg.items():
        if isinstance(metric_cfg, dict) and metric_cfg.get("enabled", False):
            # Map metric name to hook name (e.g., "f1_score" -> "f1_metric")
            hook_name = f"{metric_name.replace('_score', '')}_metric"

            if hook_name in HOOKS:
                try:
                    hook_class = HOOKS[hook_name]

                    # Remove 'enabled' from config
                    init_kwargs = {k: v for k, v in metric_cfg.items() if k != "enabled"}

                    # Instantiate metric hook
                    hook = hook_class(**init_kwargs)
                    hooks.append(hook)

                    print(f"Loaded metric hook: {metric_name} ({hook_class.__name__})")
                except Exception as e:
                    print(f"Warning: Failed to initialize metric hook '{metric_name}': {e}")
            else:
                print(
                    f"Warning: Metric hook '{hook_name}' not found in registry. "
                    f"Available: {HOOKS.registry_names}"
                )

    return hooks


def register_integration_hooks() -> None:
    """Register all integration hooks by importing their modules.

    This ensures that hook decorators are executed and hooks are
    registered in the HOOKS registry.

    Call this function before building hooks from config to ensure
    all hooks are available.
    """
    # Import integration modules to trigger registration
    try:
        from animaldet.utils.integrations import wandb  # noqa: F401
    except ImportError:
        pass

    try:
        from animaldet.utils.integrations import tensorboard  # noqa: F401
    except ImportError:
        pass
