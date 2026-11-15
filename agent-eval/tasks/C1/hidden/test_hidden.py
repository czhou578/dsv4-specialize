import pytest
from intervals import merge

def test_touch_gap_nested():
    assert merge([(3, 5), (1, 3)]) == [(1, 5)]
    assert merge([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]
    assert merge([(5, 9), (-2, 0), (6, 7), (-1, 4)]) == [(-2, 4), (5, 9)]

def test_empty():
    assert merge([]) == [] and merge([(2, 2)]) == []
    assert merge([(1, 3), (4, 4), (5, 6)]) == [(1, 3), (5, 6)]

def test_invalid_and_input():
    with pytest.raises(ValueError):
        merge([(1, 2), (5, 3)])
    data = [(3, 4), (1, 3)]
    merge(data)
    assert data == [(3, 4), (1, 3)]
