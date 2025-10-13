"""
HuggingFace Hub integration for uploading and downloading datasets.

This module provides utilities to upload animaldet datasets to HuggingFace Hub
and download them for training. Datasets are stored in their native format
(COCO format with patches + annotations or raw images + CSV annotations).
"""

import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import tempfile

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class HuggingFaceDatasetUploader:
    """
    Upload animaldet datasets to HuggingFace Hub.

    Supports two dataset formats:
    1. COCO format: train2017/, val2017/, test2017/ folders with COCO JSON annotations
    2. Processed patches: train/, val/, test/ folders with gt.csv annotations

    Args:
        dataset_dir: Path to the dataset directory
        repo_id: HuggingFace repository ID (e.g., "username/herdnet-560-all")
        token: HuggingFace API token (optional, uses HF_TOKEN env var if not provided)
        private: Whether to create a private repository
    """

    def __init__(
        self,
        dataset_dir: str,
        repo_id: str,
        token: Optional[str] = None,
        private: bool = False,
    ):
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub is required for HuggingFace integration. "
                "Install it with: pip install huggingface_hub"
            )

        self.dataset_dir = Path(dataset_dir)
        self.repo_id = repo_id
        self.token = token
        self.private = private
        self.api = HfApi(token=token)

        if not self.dataset_dir.exists():
            raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    def _detect_dataset_format(self) -> str:
        """
        Detect dataset format (coco or processed).

        Returns:
            "coco" if COCO format with train2017/val2017/test2017
            "processed" if processed patches with train/val/test + gt.csv
        """
        has_coco_format = (
            (self.dataset_dir / "train2017").exists() or
            (self.dataset_dir / "val2017").exists() or
            (self.dataset_dir / "annotations").exists()
        )

        has_processed_format = (
            (self.dataset_dir / "train").exists() and
            (self.dataset_dir / "train" / "gt.csv").exists()
        )

        if has_coco_format:
            return "coco"
        elif has_processed_format:
            return "processed"
        else:
            raise ValueError(
                f"Could not detect dataset format in {self.dataset_dir}. "
                "Expected either:\n"
                "  - COCO format: train2017/, val2017/, annotations/\n"
                "  - Processed format: train/, val/, test/ with gt.csv files"
            )

    def _collect_dataset_stats(self) -> Dict[str, Any]:
        """Collect statistics about the dataset."""
        format_type = self._detect_dataset_format()
        stats = {
            "format": format_type,
            "splits": {},
        }

        if format_type == "coco":
            # Count images in COCO format
            for split in ["train2017", "val2017", "test2017"]:
                split_dir = self.dataset_dir / split
                if split_dir.exists():
                    # Resolve symlinks
                    if split_dir.is_symlink():
                        split_dir = split_dir.resolve()

                    images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.JPG")) + \
                             list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpeg"))
                    stats["splits"][split] = {"num_images": len(images)}

            # Read COCO annotations if available
            annotations_dir = self.dataset_dir / "annotations"
            if annotations_dir.exists():
                for ann_file in annotations_dir.glob("instances_*.json"):
                    with open(ann_file) as f:
                        coco_data = json.load(f)
                        split_name = ann_file.stem.replace("instances_", "")
                        if split_name in stats["splits"]:
                            stats["splits"][split_name]["num_annotations"] = len(coco_data.get("annotations", []))
                            stats["splits"][split_name]["num_categories"] = len(coco_data.get("categories", []))

        else:  # processed format
            for split in ["train", "val", "test"]:
                split_dir = self.dataset_dir / split
                if split_dir.exists():
                    images = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.JPG")) + \
                             list(split_dir.glob("*.png")) + list(split_dir.glob("*.jpeg"))
                    stats["splits"][split] = {"num_images": len(images)}

                    # Read CSV annotations
                    csv_file = split_dir / "gt.csv"
                    if csv_file.exists():
                        import pandas as pd
                        df = pd.read_csv(csv_file)
                        stats["splits"][split]["num_annotations"] = len(df)
                        if "labels" in df.columns:
                            stats["splits"][split]["num_categories"] = df["labels"].nunique()

        return stats

    def _create_readme(self, stats: Dict[str, Any]) -> str:
        """Generate README content for the dataset."""
        readme = f"""# {self.repo_id.split('/')[-1]}

Animal detection dataset in {stats['format'].upper()} format.

## Dataset Structure

"""

        if stats["format"] == "coco":
            readme += """```
dataset/
├── train2017/          # Training images (patches)
├── val2017/            # Validation images (patches)
├── test2017/           # Test images (patches)
└── annotations/
    ├── instances_train2017.json
    ├── instances_val2017.json
    └── instances_test2017.json
```
"""
        else:
            readme += """```
dataset/
├── train/
│   ├── *.jpg          # Training patches
│   └── gt.csv         # Annotations
├── val/
│   ├── *.jpg          # Validation patches
│   └── gt.csv         # Annotations
└── test/
    ├── *.jpg          # Test patches
    └── gt.csv         # Annotations
```
"""

        readme += "\n## Dataset Statistics\n\n"
        for split, split_stats in stats["splits"].items():
            readme += f"### {split}\n"
            for key, value in split_stats.items():
                readme += f"- {key.replace('_', ' ').title()}: {value:,}\n"
            readme += "\n"

        readme += """## Usage

### Loading with HuggingFace Datasets

```python
from datasets import load_dataset

dataset = load_dataset("{}")
```

### Using with animaldet

Download and use with animaldet framework:

```bash
# Download dataset
huggingface-cli download {} --local-dir ./data/{}

# Use in training config
data:
  dataset_dir: ./data/{}
```

## Citation

If you use this dataset, please cite the original source.

## License

Please check the license of the original dataset.
""".format(self.repo_id, self.repo_id, self.repo_id.split('/')[-1], self.repo_id.split('/')[-1])

        return readme

    def _prepare_upload_directory(self) -> Path:
        """
        Prepare a temporary directory with resolved symlinks for upload.

        Returns:
            Path to temporary directory ready for upload
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="hf_upload_"))

        print(f"Preparing upload directory at {temp_dir}")

        # Copy dataset contents, resolving symlinks
        format_type = self._detect_dataset_format()

        if format_type == "coco":
            # Copy COCO format directories
            for item in ["train2017", "val2017", "test2017", "annotations"]:
                src = self.dataset_dir / item
                if src.exists():
                    dst = temp_dir / item

                    # Resolve symlink if necessary
                    if src.is_symlink():
                        src = src.resolve()

                    print(f"  Copying {item}...")
                    if src.is_dir():
                        shutil.copytree(src, dst, symlinks=False)
                    else:
                        shutil.copy2(src, dst)

        else:  # processed format
            # Copy processed format directories
            for split in ["train", "val", "test"]:
                src = self.dataset_dir / split
                if src.exists():
                    dst = temp_dir / split
                    print(f"  Copying {split}...")
                    shutil.copytree(src, dst, symlinks=False)

        return temp_dir

    def upload(
        self,
        commit_message: Optional[str] = None,
        create_readme: bool = True,
    ) -> str:
        """
        Upload dataset to HuggingFace Hub.

        Args:
            commit_message: Custom commit message (optional)
            create_readme: Whether to automatically generate a README

        Returns:
            URL of the uploaded dataset repository
        """
        print(f"{'='*80}")
        print(f"Uploading dataset to HuggingFace Hub")
        print(f"{'='*80}")
        print(f"Repository: {self.repo_id}")
        print(f"Dataset directory: {self.dataset_dir}")
        print(f"Private: {self.private}")
        print(f"{'='*80}\n")

        # Detect format and collect stats
        print("Analyzing dataset...")
        stats = self._collect_dataset_stats()
        print(f"Format: {stats['format']}")
        print(f"Splits: {', '.join(stats['splits'].keys())}\n")

        # Create repository
        print("Creating repository on HuggingFace Hub...")
        try:
            create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=self.private,
                token=self.token,
                exist_ok=True,
            )
            print(f"✓ Repository created/verified: {self.repo_id}\n")
        except Exception as e:
            raise RuntimeError(f"Failed to create repository: {e}")

        # Prepare upload directory (resolve symlinks)
        upload_dir = self._prepare_upload_directory()

        try:
            # Create README if requested
            if create_readme:
                print("Generating README...")
                readme_content = self._create_readme(stats)
                readme_path = upload_dir / "README.md"
                readme_path.write_text(readme_content)
                print("✓ README generated\n")

            # Create dataset metadata
            metadata = {
                "format": stats["format"],
                "splits": list(stats["splits"].keys()),
                "stats": stats,
            }
            metadata_path = upload_dir / ".dataset_info.json"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            # Upload to Hub
            print("Uploading files to HuggingFace Hub...")
            print("This may take a while depending on dataset size...\n")

            if commit_message is None:
                commit_message = f"Upload {stats['format']} format dataset"

            upload_folder(
                repo_id=self.repo_id,
                folder_path=str(upload_dir),
                repo_type="dataset",
                token=self.token,
                commit_message=commit_message,
            )

            repo_url = f"https://huggingface.co/datasets/{self.repo_id}"

            print(f"\n{'='*80}")
            print(f"✓ Dataset uploaded successfully!")
            print(f"{'='*80}")
            print(f"Repository URL: {repo_url}")
            print(f"{'='*80}\n")

            return repo_url

        finally:
            # Cleanup temporary directory
            print("Cleaning up temporary files...")
            shutil.rmtree(upload_dir, ignore_errors=True)
            print("✓ Cleanup complete\n")


def upload_dataset_to_hf(
    dataset_dir: str,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: Optional[str] = None,
    create_readme: bool = True,
) -> str:
    """
    Upload a dataset to HuggingFace Hub.

    Convenience function for uploading datasets.

    Args:
        dataset_dir: Path to the dataset directory
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        token: HuggingFace API token (optional)
        private: Whether to create a private repository
        commit_message: Custom commit message (optional)
        create_readme: Whether to automatically generate a README

    Returns:
        URL of the uploaded dataset repository

    Example:
        >>> upload_dataset_to_hf(
        ...     dataset_dir="data/herdnet/processed/560_all",
        ...     repo_id="myorg/herdnet-560-all",
        ...     private=False
        ... )
    """
    uploader = HuggingFaceDatasetUploader(
        dataset_dir=dataset_dir,
        repo_id=repo_id,
        token=token,
        private=private,
    )

    return uploader.upload(
        commit_message=commit_message,
        create_readme=create_readme,
    )
