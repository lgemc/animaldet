# Dataset information and sources:

## Herdnet dataset

Presented at: [Multispecies detection and identification of African mammals in aerial imagery using convolutional neural networks](https://zslpublications.onlinelibrary.wiley.com/doi/full/10.1002/rse2.234)

Sources extracted from: [Uliege University repository](https://dataverse.uliege.be/dataset.xhtml?persistentId=doi:10.58119/ULG/MIRUU5)

MD5: 989b5ab2a37ead9a5b5df6a5a5aa64a1

Folder in the current project: `data/herdnet/raw/...`

Folder structure:

```txt
└── ./
    ├── groundtruth/
    │   ├── csv/
    │   │   ├── test_big_size_A_B_E_K_WH_WB.csv
    │   │   ├── train_big_size_A_B_E_K_WH_WB.csv
    │   │   └── val_big_size_A_B_E_K_WH_WB.csv
    │   └── json/
    │       ├── test_subframes_A_B_E_K_WH_WB.json
    │       ├── train_subframes_A_B_E_K_WH_WB.json
    │       └── val_subframes_A_B_E_K_WH_WB.json*
    ├── test/
    │   └── *.JPG
    ├── train/
    │   └── *.JPG
    ├── train_subframes/
    │   └── *.JPG
    └── val/
        └── *.JPG
```

## Delplanque uploaded dataset

Presented at: [HerdNet repo](https://github.com/Alexandre-Delplanque/HerdNet)

As described at [demo notebook](https://github.com/Alexandre-Delplanque/HerdNet/blob/main/notebooks/demo-training-testing-herdnet.ipynb), at
drive there is a modified dataset from original `Multispecies detection`, uploaded to drive:

```bash
gdown 1CcTAZZJdwrBfCPJtVH6VBU3luGKIN9st -O /content/data.zip
```

MD5: a1b8680585c286740864a91a1b305f20

Folder in the current project: `data/herdnet/processed/delplanque...`

```txt
.
├── test/
│   └── *.JPG
├── test.csv
├── train/
│   └── *.JPG
├── train.csv
├── train_patches/
│   └── *.JPG
├── train_patches.csv
├── val/
│   └── *.JPG
└── val.csv
```