from intervals import merge

def test_basic():
    assert merge([(1, 4), (2, 6)]) == [(1, 6)]
    assert merge([(7, 9), (1, 3)]) == [(1, 3), (7, 9)]
