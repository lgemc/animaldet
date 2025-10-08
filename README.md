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