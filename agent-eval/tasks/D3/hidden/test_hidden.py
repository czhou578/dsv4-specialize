from catalog import Catalog

def test_all_methods_forget():
    c = Catalog({"a": 2.0, "b": 1.0})
    assert (c.unit("a"), c.quote("a", 3), c.quote("b", 20)) == (2.0, 6.0, 18.0)
    c.set_price("a", 7.0)
    assert (c.unit("a"), c.quote("a", 3), c.quote("b", 20)) == (7.0, 21.0, 18.0)
    c.set_price("b", 2.0)
    assert (c.unit("b"), c.quote("b", 20)) == (2.0, 36.0)

def test_memo_still_caches():
    c = Catalog({"a": 1.0})
    c.quote("a", 1)
    c.prices["a"] = 9.0
    assert c.quote("a", 1) == 1.0
