import sys

args = sys.argv

if len(args) != 7:
    print("Usage: python 4_eval_test_scores.py <work_dir> <test_csv> <test_root_dir> <pth_path> <detections_output> <results_output>")
    sys.exit(1)

import os
from animaloc.utils.seed import set_seed
import albumentations as A
from animaloc.datasets import CSVDataset
from animaloc.data.transforms import  DownSample
from animaloc.models import HerdNet, LossWrapper, load_model
from torch.utils.data import DataLoader
from animaloc.eval import PointsMetrics, HerdNetStitcher, HerdNetEvaluator
import torch

set_seed(9292)

patch_size = 512
num_classes = 7
down_ratio = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

work_dir = args[1]  # output directory, example: '/workspace/data'
os.makedirs(work_dir, exist_ok=True)

test_dataset = CSVDataset(
    csv_file = args[2],  # example: '/workspace/data/test.csv',
    root_dir = args[3],  # example: '/workspace/data/test',
    albu_transforms = [A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )],
    end_transforms = [DownSample(down_ratio=down_ratio, anno_type='point')]
)

test_dataloader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = False)

herdnet = HerdNet(num_classes=num_classes, down_ratio=down_ratio).to(device)
herdnet = LossWrapper(herdnet, losses=[])
herdnet = load_model(herdnet, pth_path=args[4])  # example: "/workspace/data/last_model_stage_2.pth"


metrics = PointsMetrics(radius=5, num_classes=num_classes) # radius for herdnet data

stitcher = HerdNetStitcher(
    model=herdnet,
    size=(patch_size,patch_size),
    overlap=0,
    down_ratio=down_ratio,
    reduction='mean',
    up=False
)

evaluator = HerdNetEvaluator(
    model=herdnet,
    dataloader=test_dataloader,
    metrics=metrics,
    stitcher=stitcher,
    work_dir=work_dir,
    header='validation',
    print_freq=10,
    lmds_kwargs={
        "kernel_size": (3,3),
        "adapt_ts": 0.3,
    }
)

test_f1_score = evaluator.evaluate(returns='f1_score')
print(f"F1 score = {test_f1_score * 100:0.0f}%")
results = evaluator.results
results.to_csv(args[5], index=False)  # example: "/workspace/data/test_detections.csv"
detections=evaluator.detections
detections.to_csv(args[6], index=False)  # example: "/workspace/data/test_results.csv"