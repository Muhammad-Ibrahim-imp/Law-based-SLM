import torch

def validate_batch(batch):
    if not isinstance(batch, dict):
        raise TypeError("batch must be a dict")

    required_keys = {
        "input_ids",
        "labels",
    }
    missing_keys = required_keys - batch.keys()

    if missing_keys:
        raise KeyError("Missing keys in batch: {}".format(missing_keys))

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")

    if input_ids.ndim != 2:
        raise ValueError("input_ids must be a 2d array")
    if labels.ndim != 2:
        raise ValueError("labels must be a 2d array")

    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape")

    return True



