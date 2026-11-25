"""Exercise a Python application both with and without the panel profile."""

import os

import torch
import torch.nn.functional as functional


enabled=os.environ.get('PU_ENABLED')=='1'
assert (functional.linear.__module__=='pascal_unleashed.torch')==enabled
with torch.inference_mode():
    a=torch.ones((128,256),device='cuda',dtype=torch.float16)
    b=torch.ones((256,256),device='cuda',dtype=torch.float16)
    out=functional.linear(a,b)
    assert out.shape==(128,256)
    assert torch.max(torch.abs(out.float()-256)).item()<2
