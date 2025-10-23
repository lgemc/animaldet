#!/usr/bin/env python3
"""
Sample a percentage of patches from COCO annotations while keeping all animals in selected patches.
Uses stratified sampling to maintain class distributions.

Usage:
    uv run scripts/sample_patches.py <input_json> <output_json> --percentage <percentage>
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Set
from collections import defaultdict
import argparse


def calculate_class_distribution(
    annotations: List[Dict],
    categories: List[Dict]
) -> Dict[int, Dict[str, Any]]:
    """Calculate distribution statistics for each class."""
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # Count annotations per class
    class_counts = defaultdict(int)
    for ann in annotations:
        class_counts[ann['category_id']] += 1

    # Calculate statistics
    total_annotations = len(annotations)
    distribution = {}

    for cat_id, count in class_counts.items():
        distribution[cat_id] = {
            'name': cat_id_to_name[cat_id],
            'count': count,
            'percentage': (count / total_annotations * 100) if total_annotations > 0 else 0
        }

    return distribution


def compare_distributions(
    original_annotations: List[Dict],
    sampled_annotations: List[Dict],
    categories: List[Dict]
) -> None:
    """Compare and print distribution statistics."""
    print("\n" + "="*80)
    print("CLASS DISTRIBUTION COMPARISON")
    print("="*80)

    original_dist = calculate_class_distribution(original_annotations, categories)
    sampled_dist = calculate_class_distribution(sampled_annotations, categories)

    # Print header
    print(f"\n{'Class':<20} {'Original':<25} {'Sampled':<25} {'Diff':<10}")
    print(f"{'Name':<20} {'Count':<10} {'%':<15} {'Count':<10} {'%':<15} {'%':<10}")
    print("-" * 80)

    # Print each class
    for cat_id in sorted(original_dist.keys()):
        orig = original_dist[cat_id]
        samp = sampled_dist.get(cat_id, {'name': orig['name'], 'count': 0, 'percentage': 0})

        diff = samp['percentage'] - orig['percentage']

        print(f"{orig['name']:<20} "
              f"{orig['count']:<10} {orig['percentage']:<14.2f}% "
              f"{samp['count']:<10} {samp['percentage']:<14.2f}% "
              f"{diff:>+9.2f}%")

    # Print totals
    print("-" * 80)
    print(f"{'TOTAL':<20} "
          f"{len(original_annotations):<10} {100.0:<14.2f}% "
          f"{len(sampled_annotations):<10} {100.0:<14.2f}%")
    print("="*80 + "\n")


def build_image_to_categories(
    images: List[Dict],
    annotations: List[Dict]
) -> Dict[int, Set[int]]:
    """Build mapping from image_id to set of category_ids present."""
    image_to_cats = defaultdict(set)

    for ann in annotations:
        image_to_cats[ann['image_id']].add(ann['category_id'])

    # Ensure all images are in the dict, even those without annotations
    for img in images:
        if img['id'] not in image_to_cats:
            image_to_cats[img['id']] = set()

    return image_to_cats


def stratified_sample_patches(
    images: List[Dict],
    image_to_categories: Dict[int, Set[int]],
    num_samples: int,
    seed: int
) -> List[Dict]:
    """
    Stratified sampling of patches to maintain class distribution.

    Strategy: Group patches by their category combination (multi-label),
    then sample proportionally from each stratum.
    """
    random.seed(seed)

    # Group images by their category combinations
    strata = defaultdict(list)
    for img in images:
        # Create a hashable key from the set of categories
        cat_tuple = tuple(sorted(image_to_categories[img['id']]))
        strata[cat_tuple].append(img)

    print(f"\nStratification:")
    print(f"  Total strata (unique category combinations): {len(strata)}")
    print(f"  Largest stratum: {max(len(imgs) for imgs in strata.values())} patches")
    print(f"  Smallest stratum: {min(len(imgs) for imgs in strata.values())} patches")

    # Calculate samples per stratum
    total_images = len(images)
    sampled_images = []

    for cat_tuple, stratum_images in strata.items():
        # Calculate proportion of this stratum
        stratum_proportion = len(stratum_images) / total_images

        # Calculate how many samples from this stratum
        stratum_samples = max(1, round(num_samples * stratum_proportion))

        # Don't sample more than available
        stratum_samples = min(stratum_samples, len(stratum_images))

        # Sample from this stratum
        sampled = random.sample(stratum_images, stratum_samples)
        sampled_images.extend(sampled)

    # If we have more samples than needed (due to rounding), randomly remove some
    if len(sampled_images) > num_samples:
        sampled_images = random.sample(sampled_images, num_samples)

    # If we have fewer samples than needed (rare), add more randomly
    while len(sampled_images) < num_samples:
        remaining = [img for img in images if img not in sampled_images]
        if not remaining:
            break
        sampled_images.append(random.choice(remaining))

    return sampled_images


def sample_patches(
    input_path: str,
    output_path: str,
    percentage: float,
    seed: int = 42,
    use_stratified: bool = True
) -> None:
    """
    Sample a percentage of patches from COCO annotations.

    Args:
        input_path: Path to input COCO JSON file
        output_path: Path to output COCO JSON file
        percentage: Percentage of patches to sample (0-100)
        seed: Random seed for reproducibility
        use_stratified: Use stratified sampling to maintain distribution
    """
    # Validate percentage
    if not 0 < percentage <= 100:
        raise ValueError(f"Percentage must be between 0 and 100, got {percentage}")

    # Load annotations
    print(f"Loading annotations from {input_path}...")
    with open(input_path, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    print(f"\nDataset Statistics:")
    print(f"  Total patches (images): {len(images)}")
    print(f"  Total animals (annotations): {len(annotations)}")
    print(f"  Categories: {len(categories)}")

    # Calculate number of patches to sample
    num_samples = int(len(images) * (percentage / 100))
    print(f"\nSampling Strategy:")
    print(f"  Method: {'Stratified' if use_stratified else 'Random'}")
    print(f"  Target: {percentage}% = {num_samples} patches")

    # Build image to categories mapping
    image_to_categories = build_image_to_categories(images, annotations)

    # Sample patches
    if use_stratified:
        sampled_images = stratified_sample_patches(
            images, image_to_categories, num_samples, seed
        )
    else:
        random.seed(seed)
        sampled_images = random.sample(images, num_samples)

    sampled_image_ids = {img['id'] for img in sampled_images}

    # Filter annotations to only include those for sampled images
    sampled_annotations = [
        ann for ann in annotations
        if ann['image_id'] in sampled_image_ids
    ]

    print(f"\nSampling Results:")
    print(f"  Selected patches: {len(sampled_images)}")
    print(f"  Selected animals: {len(sampled_annotations)}")
    print(f"  Average animals per patch: {len(sampled_annotations) / len(sampled_images):.2f}")

    # Compare distributions
    compare_distributions(annotations, sampled_annotations, categories)

    # Create output COCO data
    output_data = {
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': categories
    }

    # Add info if present
    if 'info' in coco_data:
        output_data['info'] = coco_data['info']

    # Save output
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving sampled annotations to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output_data, f)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Sample a percentage of patches from COCO annotations with stratified sampling"
    )
    parser.add_argument(
        "input_json",
        type=str,
        help="Path to input COCO JSON file"
    )
    parser.add_argument(
        "output_json",
        type=str,
        help="Path to output COCO JSON file"
    )
    parser.add_argument(
        "--percentage",
        type=float,
        required=True,
        help="Percentage of patches to sample (0-100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-stratified",
        action="store_true",
        help="Disable stratified sampling (use random sampling instead)"
    )

    args = parser.parse_args()

    sample_patches(
        input_path=args.input_json,
        output_path=args.output_json,
        percentage=args.percentage,
        seed=args.seed,
        use_stratified=not args.no_stratified
    )


if __name__ == "__main__":
    main()
