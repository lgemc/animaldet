"""Visualization command implementation."""

import os
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf

from animaldet.visualization import load_herdnet_dataset, load_ungulate_dataset


def visualize_main(
    config: Optional[str],
    dataset_type: Optional[str],
    csv_path: Optional[str],
    images_dir: Optional[str],
    name: str,
    persistent: bool,
    port: int,
    remote: bool,
    max_samples: Optional[int],
):
    """Launch FiftyOne visualization for animaldet datasets.

    Args:
        config: Path to configuration file
        dataset_type: Dataset type (ungulate or herdnet)
        csv_path: Path to CSV file with annotations
        images_dir: Directory containing images
        name: Dataset name in FiftyOne
        persistent: Persist dataset to FiftyOne database
        port: Port for FiftyOne app
        remote: Launch in remote mode (for SSH sessions)
        max_samples: Maximum number of samples to load (for debugging)
    """
    # Load config if provided
    if config:
        cfg = OmegaConf.load(config)
        dataset_type = dataset_type or cfg.dataset_type
        csv_path = csv_path or cfg.csv_path
        images_dir = images_dir or cfg.images_dir
        name = cfg.get("name", name)
        persistent = cfg.get("persistent", persistent)
        database_uri = cfg.get("database_uri", None)
        max_samples = max_samples or cfg.get("max_samples", None)
    else:
        # Use CLI arguments
        if not dataset_type or not csv_path or not images_dir:
            raise ValueError(
                "--dataset-type, --csv-path, and --images-dir are required when --config is not provided"
            )
        database_uri = None

    # Set database URI as environment variable BEFORE importing fiftyone
    # This ensures it's picked up during initialization
    if database_uri:
        os.environ["FIFTYONE_DATABASE_URI"] = database_uri
        print(f"Configured FiftyOne to use MongoDB at: {database_uri}")

    # Import fiftyone AFTER setting environment variable
    import fiftyone as fo

    # Convert to absolute paths
    csv_path = Path(csv_path).absolute()
    images_dir = Path(images_dir).absolute()

    # Load dataset
    print(f"Loading {dataset_type} dataset from {csv_path}")
    if max_samples:
        print(f"Limiting to {max_samples} samples for debugging")
    if dataset_type == "ungulate":
        dataset = load_ungulate_dataset(
            csv_path=csv_path,
            images_dir=images_dir,
            name=name,
            persistent=persistent,
            database_uri=database_uri,
            max_samples=max_samples,
        )
    elif dataset_type == "herdnet":
        dataset = load_herdnet_dataset(
            csv_path=csv_path,
            images_dir=images_dir,
            name=name,
            persistent=persistent,
            database_uri=database_uri,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Launch FiftyOne app
    print(f"\nLaunching FiftyOne app on port {port}...")
    print(f"Dataset: {name} ({len(dataset)} samples)")

    session = fo.launch_app(dataset, port=port, remote=remote)

    print(f"\n✓ FiftyOne app launched at http://localhost:{port}")
    print("Press Ctrl+C to stop the app")

    # Keep session alive
    session.wait()