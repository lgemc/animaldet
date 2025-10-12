#!/usr/bin/env python3
"""
Centralized CLI for animaldet tools.

This CLI provides a unified interface for all animaldet tools including:
- train: Train animal detection models
- patcher: Extract patches from images
- visualize: Visualize datasets with FiftyOne

Usage:
    animaldet train --config configs/experiment/herdnet.yaml
    animaldet patcher --config configs/patcher/default.yaml
    animaldet visualize --dataset-type herdnet --csv-path data.csv --images-dir images/
"""

import typer
from typing_extensions import Annotated

app = typer.Typer(
    name="animaldet",
    help="Animaldet CLI - Tools for animal detection tasks",
    add_completion=False,
    no_args_is_help=True,
)


@app.command(name="train")
def train(
    config: Annotated[str, typer.Option("--config", help="Path to experiment configuration file (YAML)")] = None,
    trainer: Annotated[str, typer.Option("--trainer", help="Trainer name from registry (overrides config)")] = None,
    work_dir: Annotated[str, typer.Option("--work-dir", help="Working directory for outputs (overrides config)")] = None,
    seed: Annotated[int, typer.Option("--seed", help="Random seed (overrides config)")] = None,
    resume: Annotated[str, typer.Option("--resume", help="Path to checkpoint to resume training from")] = None,
    device: Annotated[str, typer.Option("--device", help="Device to use for training")] = "cuda",
    debug: Annotated[bool, typer.Option("--debug", help="Enable debug mode with verbose logging")] = False,
):
    """Train animal detection models."""
    from tools.train_cmd import train_main

    train_main(
        config=config,
        trainer=trainer,
        work_dir=work_dir,
        seed=seed,
        resume=resume,
        device=device,
        debug=debug,
    )


@app.command(name="patcher")
def patcher(
    config: Annotated[str, typer.Option("--config", help="Path to patcher configuration file (YAML)")] = None,
    images_root: Annotated[str, typer.Option("--images-root", help="Root directory containing images")] = None,
    dest_dir: Annotated[str, typer.Option("--dest-dir", help="Destination directory for patches")] = None,
    patch_size: Annotated[int, typer.Option("--patch-size", help="Size of patches to extract")] = None,
    overlap: Annotated[int, typer.Option("--overlap", help="Overlap between patches in pixels")] = 0,
    csv_path: Annotated[str, typer.Option("--csv-path", help="Path to CSV file with annotations")] = None,
    save_all: Annotated[bool, typer.Option("--save-all", help="Save all patches including those without annotations")] = False,
):
    """Extract patches from images using configuration or CLI arguments."""
    from tools.data.patcher import patcher_main

    patcher_main(
        config=config,
        images_root=images_root,
        dest_dir=dest_dir,
        patch_size=patch_size,
        overlap=overlap,
        csv_path=csv_path,
        save_all=save_all,
    )


@app.command(name="visualize")
def visualize(
    config: Annotated[str, typer.Option("--config", help="Path to visualization config YAML file")] = None,
    dataset_type: Annotated[str, typer.Option("--dataset-type", help="Dataset type (ungulate or herdnet)")] = None,
    csv_path: Annotated[str, typer.Option("--csv-path", help="Path to CSV file with annotations")] = None,
    gt_csv_path: Annotated[str, typer.Option("--gt-csv-path", help="Path to CSV file with ground truth annotations (optional)")] = None,
    images_dir: Annotated[str, typer.Option("--images-dir", help="Directory containing images")] = None,
    name: Annotated[str, typer.Option("--name", help="Dataset name in FiftyOne")] = "dataset",
    persistent: Annotated[bool, typer.Option("--persistent", help="Persist dataset to FiftyOne database")] = False,
    port: Annotated[int, typer.Option("--port", help="Port for FiftyOne app")] = 5151,
    remote: Annotated[bool, typer.Option("--remote", help="Launch in remote mode (for SSH sessions)")] = False,
    max_samples: Annotated[int, typer.Option("--max-samples", help="Maximum number of samples to load (for debugging)")] = None,
):
    """Visualize datasets with FiftyOne."""
    from tools.data.visualization import visualize_main

    visualize_main(
        config=config,
        dataset_type=dataset_type,
        csv_path=csv_path,
        gt_csv_path=gt_csv_path,
        images_dir=images_dir,
        name=name,
        persistent=persistent,
        port=port,
        remote=remote,
        max_samples=max_samples,
    )


@app.command(name="infer")
def infer(
    config: Annotated[str, typer.Option("--config", help="Path to inference configuration file (YAML)")] = None,
    checkpoint: Annotated[str, typer.Option("--checkpoint", help="Path to model checkpoint")] = None,
    images_dir: Annotated[str, typer.Option("--images-dir", help="Directory containing images")] = None,
    output_csv: Annotated[str, typer.Option("--output-csv", help="Path to output CSV file")] = None,
    threshold: Annotated[float, typer.Option("--threshold", help="Detection threshold")] = None,
    device: Annotated[str, typer.Option("--device", help="Device to use for inference")] = "cuda",
    batch_size: Annotated[int, typer.Option("--batch-size", help="Batch size for inference")] = None,
    model_type: Annotated[str, typer.Option("--model-type", help="Model type: 'herdnet' or 'rfdetr' (auto-detected from config if not specified)")] = None,
):
    """Run inference on full-size images using stitcher (supports HerdNet and RF-DETR)."""
    from tools.inference_cmd import inference_main

    inference_main(
        config=config,
        checkpoint=checkpoint,
        images_dir=images_dir,
        output_csv=output_csv,
        threshold=threshold,
        device=device,
        batch_size=batch_size,
        model_type=model_type,
    )


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()