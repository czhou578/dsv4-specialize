from billing import invoice
from order import Order

def test_new_orders_start_empty():
    Order("a").add("x", 1, 1.0)
    assert Order("b").items == []
    assert Order("c").items is not Order("d").items

def test_given_items_are_kept():
    o = Order("e", [("y", 2, 3.0)]).add("z", 1, 1.0)
    assert o.items == [("y", 2, 3.0), ("z", 1, 1.0)]

def test_many_invoices():
    for i in range(1, 4):
        assert invoice(f"c{i}", [("pen", i, 1.0)])["lines"] == 1
    assert invoice("f", [("a", 1, 2.0), ("b", 3, 1.0)])["total"] == 5.4
