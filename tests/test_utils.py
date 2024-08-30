import torch
import numpy as np

from cata2data import collate_fn_irregular_cutouts, collate_fn_regular_cutouts

def test_collate_fn_irregular_cutouts_with_arraylike_y():
    batch = [
        (np.random.rand(3, 4), np.array([1, 2, 3])),
        (np.random.rand(3, 4), np.array([4, 5, 6])),
    ]
    x, y = collate_fn_irregular_cutouts(batch)

    assert isinstance(x, list), "Expected x to be a list."
    assert isinstance(y, torch.Tensor), "Expected y to be a tensor."
    assert len(x) == 2, "Expected x to have length 2."
    assert y.shape == (6,), "Expected y to be concatenated correctly."

def test_collate_fn_irregular_cutouts_with_scalar_y():
    batch = [
        (np.random.rand(3, 4), 1),
        (np.random.rand(3, 4), 0),
    ]
    x, y = collate_fn_irregular_cutouts(batch)

    assert isinstance(x, list), "Expected x to be a list."
    assert isinstance(y, torch.Tensor), "Expected y to be a tensor."
    assert len(x) == 2, "Expected x to have length 2."
    assert y.shape == (2,), "Expected y to have shape (2,)."
    assert torch.equal(y, torch.tensor([1, 0])), "Expected y to match the input scalar values."

def test_collate_fn_regular_cutouts_with_arraylike_y():
    batch = [
        (torch.rand(3, 4), np.array([1, 2, 3])),
        (torch.rand(3, 4), np.array([4, 5, 6])),
    ]
    x, y = collate_fn_regular_cutouts(batch)

    assert isinstance(x, torch.Tensor), "Expected x to be a tensor."
    assert isinstance(y, torch.Tensor), "Expected y to be a tensor."
    assert x.shape == (2, 3, 4), "Expected x to have shape (2, 3, 4)."
    assert y.shape == (6,), "Expected y to be concatenated correctly."

def test_collate_fn_regular_cutouts_with_scalar_y():
    batch = [
        (torch.rand(3, 4), 1),
        (torch.rand(3, 4), 0),
    ]
    x, y = collate_fn_regular_cutouts(batch)

    assert isinstance(x, torch.Tensor), "Expected x to be a tensor."
    assert isinstance(y, torch.Tensor), "Expected y to be a tensor."
    assert x.shape == (2, 3, 4), "Expected x to have shape (2, 3, 4)."
    assert y.shape == (2,), "Expected y to have shape (2,)."
    assert torch.equal(y, torch.tensor([1, 0])), "Expected y to match the input scalar values."

def test_collate_fn_irregular_cutouts_empty_batch():
    batch = []
    x, y = collate_fn_irregular_cutouts(batch)

    assert x.numel() == 0, "Expected x to be an empty tensor."
    assert y.numel() == 0, "Expected y to be an empty tensor."

def test_collate_fn_regular_cutouts_empty_batch():
    batch = []
    x, y = collate_fn_regular_cutouts(batch)

    assert x.numel() == 0, "Expected x to be an empty tensor."
    assert y.numel() == 0, "Expected y to be an empty tensor."
