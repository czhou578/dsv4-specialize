import pytest
from lru import LRUCache

def test_capacity():
    with pytest.raises(ValueError):
        LRUCache(-1)
    c = LRUCache(0)
    c.put("a", 1)
    assert len(c) == 0 and c.get("a", 7) == 7 and c.keys() == []

def test_order():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    c.put("a", 10)
    assert c.keys() == ["b", "a"] and c.get("a") == 10
    c.put("c", 3)
    assert c.keys() == ["a", "c"]
    assert c.peek("a") == 10 and c.peek("z", 0) == 0 and c.get("z") is None
    c.put("d", 4)
    assert c.keys() == ["c", "d"]

def test_none_value():
    c = LRUCache(1)
    c.put("a", None)
    assert c.get("a", 5) is None and len(c) == 1
