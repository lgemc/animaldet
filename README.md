# Animal detection 

Animal detection is a framework to handle datasets, models and training for animal detection tasks.

# Data tools

The `tools/data/patcher` allows to patch large images into smaller crops.

```bash
uv run tools/patcher.py \
    images_root=./data/herdnet/images \
    dest_dir=./data/patches \
    patch_size=512 \
    overlap=160 \
    save_all=true
```

# Visualizations

We use [fiftyone](https://github.com/voxel51/fiftyone) to visualize the datasets, using the next command:

```bash
uv run tools/main.py visualize --config configs/visualization/herdnet/raw/train.yaml
```