import pytest
import torch

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this machine"
)
def test_cuda_is_available():
    assert torch.cuda.is_available()

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available on this machine"
)
def test_bfloat16_suppported():
    assert torch.cuda.is_bf16_supported()