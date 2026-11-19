from lru import LRUCache

def test_basic():
    c = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)
    assert c.get("b") is None and c.keys() == ["a", "c"] and len(c) == 2
