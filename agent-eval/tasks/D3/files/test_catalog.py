from catalog import Catalog

def test_quote_after_price_change():
    c = Catalog({"tea": 4.0, "jam": 3.0})
    assert c.quote("tea", 10) == 36.0
    c.set_price("tea", 5.0)
    assert c.quote("tea", 10) == 45.0
