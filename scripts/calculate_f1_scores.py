#!/usr/bin/env python3
"""
Script to calculate F1 scores for detection results.

Usage:
    python scripts/calculate_f1_scores.py configs/metrics/f1_rfdetr.yaml
    python scripts/calculate_f1_scores.py configs/metrics/f1_herdnet.yaml
"""

import hydra
from omegaconf import DictConfig
import logging
from animaldet.evaluation.calculate_f1 import calculate_f1_scores

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs/metrics", config_name="f1_rfdetr")
def main(cfg: DictConfig):
    """Main function to calculate F1 scores."""
    logger.info("Starting F1 score calculation")
    logger.info(f"Ground truth: {cfg.gt_csv_path}")
    logger.info(f"Predictions: {cfg.pred_csv_path}")
    logger.info(f"Output: {cfg.output_csv_path}")
    logger.info(f"Distance threshold: {cfg.distance_threshold}px")

    results_df = calculate_f1_scores(
        gt_csv_path=cfg.gt_csv_path,
        pred_csv_path=cfg.pred_csv_path,
        output_csv_path=cfg.output_csv_path,
        prediction_format=cfg.prediction_format,
        distance_threshold=cfg.distance_threshold
    )

    logger.info(f"F1 score calculation complete. Results saved to {cfg.output_csv_path}")


if __name__ == "__main__":
    main()