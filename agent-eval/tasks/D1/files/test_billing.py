from billing import invoice

def test_invoice_ann():
    assert invoice("ann", [("pen", 2, 1.50)])["total"] == 3.24

def test_invoice_bob():
    assert invoice("bob", [("ink", 1, 10.00)]) == {"customer": "bob", "lines": 1, "total": 10.8}
