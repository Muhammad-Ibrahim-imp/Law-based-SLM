import json

import numpy as np

from data.shard_dataset import ShardedTokenDataset


def test_dataset_yields_correct_shape(tmp_path):
    category = "general_train"
    category_dir = tmp_path / category
    category_dir.mkdir()
    tokens = np.arange(2050, dtype=np.uint16)
    tokens.tofile(category_dir / "shard_00000.bin")
    (tmp_path / "manifest.json").write_text(
        json.dumps({
            "categories": {
                category: {
                    "shards": [{"file": "shard_00000.bin", "tokens": len(tokens)}]
                }
            }
        }),
        encoding="utf-8",
    )
    ds = ShardedTokenDataset(
        dataset_root=str(tmp_path),
        category=category,
        sequence_length=2048,
    )
    item = ds[0]
    assert item["input_ids"].shape == (2048,)
    assert item["labels"].shape == (2048,)
    assert item["input_ids"].dtype.is_floating_point is False  # torch.long
    # labels should be input_ids shifted by one
    assert (item["labels"][:-1] == item["input_ids"][1:]).all()