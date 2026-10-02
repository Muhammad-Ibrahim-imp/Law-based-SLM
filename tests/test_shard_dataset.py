# tests/test_shard_dataset.py 
from data.shard_dataset import ShardedTokenDataset

def test_dataset_yields_correct_shape():
    ds = ShardedTokenDataset(dataset_root="./dataset", category="general", sequence_length=2048)
    item = ds[0]
    assert item["input_ids"].shape == (2048,)
    assert item["labels"].shape == (2048,)
    assert item["input_ids"].dtype.is_floating_point is False  # torch.long
    # labels should be input_ids shifted by one
    assert (item["labels"][:-1] == item["input_ids"][1:]).all()