import pytest
import torch

from training.batch import validate_batch

def test_valid_batch():
    batch = {
    "input_ids" : torch.randint(0, 100, (2,32)),
    "labels" : torch.randint(0, 100, (2,32))
    }
    assert validate_batch(batch) is True

def test_batch_must_be_dictionary():
    with pytest.raises(TypeError):
        validate_batch(torch.randint(0, 100, (2,32)))

def test_missing_input_ids():
    batch = {
    "labels" : torch.randint(0, 100, (2,32))
    }
    with pytest.raises(KeyError):
        validate_batch(batch)

def test_missing_labels():
    batch = {
    "input_ids" : torch.randint(0, 100, (2,32)),
    }
    with pytest.raises(KeyError):
        validate_batch(batch)

def test_input_ids_dim():
    batch = {
    "input_ids" : torch.randint(0, 100, (2,32,1)),
    "labels" : torch.randint(0, 100, (2,32))
    }
    with pytest.raises(ValueError):
        validate_batch(batch)

def test_labels_dim():
    batch = {
    "input_ids" : torch.randint(0, 100, (2,32)),
    "labels" : torch.randint(0, 100, (2,32,1))
    }
    with pytest.raises(ValueError):
        validate_batch(batch)

def test_inputs_and_labels_same_shape():
    batch = {
    "input_ids" : torch.randint(0, 100, (2,32)),
    "labels" : torch.randint(0, 100, (2,31))
    }
    with pytest.raises(ValueError):
        validate_batch(batch)