"""CLI tool to visualize datasets with FiftyOne."""

import argparse
from pathlib import Path

import fiftyone as fo
from omegaconf import OmegaConf

from animaldet.visualization import load_herdnet_dataset, load_ungulate_dataset


def main():
    """Launch FiftyOne visualization for animaldet datasets."""
    parser = argparse.ArgumentParser(
        description="Visualize animaldet datasets with FiftyOne"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to visualization config YAML file",
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["ungulate", "herdnet"],
        help="Dataset type (ungulate or herdnet)",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        help="Path to CSV file with annotations",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        help="Directory containing images",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="dataset",
        help="Dataset name in FiftyOne",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        help="Persist dataset to FiftyOne database",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for FiftyOne app (default: 5151)",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Launch in remote mode (for SSH sessions)",
    )

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        config = OmegaConf.load(args.config)
        dataset_type = config.dataset_type
        csv_path = config.csv_path
        images_dir = config.images_dir
        name = config.name
        persistent = config.get("persistent", False)
    else:
        # Use CLI arguments
        if not args.dataset_type or not args.csv_path or not args.images_dir:
            parser.error(
                "--dataset-type, --csv-path, and --images-dir are required when --config is not provided"
            )

        dataset_type = args.dataset_type
        csv_path = args.csv_path
        images_dir = args.images_dir
        name = args.name
        persistent = args.persistent

    # Convert to absolute paths
    csv_path = Path(csv_path).absolute()
    images_dir = Path(images_dir).absolute()

    # Load dataset
    print(f"Loading {dataset_type} dataset from {csv_path}")
    if dataset_type == "ungulate":
        dataset = load_ungulate_dataset(
            csv_path=csv_path,
            images_dir=images_dir,
            name=name,
            persistent=persistent,
        )
    elif dataset_type == "herdnet":
        dataset = load_herdnet_dataset(
            csv_path=csv_path,
            images_dir=images_dir,
            name=name,
            persistent=persistent,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    # Launch FiftyOne app
    print(f"\nLaunching FiftyOne app on port {args.port}...")
    print(f"Dataset: {name} ({len(dataset)} samples)")

    session = fo.launch_app(dataset, port=args.port, remote=args.remote)

    print(f"\n✓ FiftyOne app launched at http://localhost:{args.port}")
    print("Press Ctrl+C to stop the app")

    # Keep session alive
    session.wait()


if __name__ == "__main__":
    main()
