from order import Order

TAX = 0.08

def subtotal(order):
    return sum(qty * price for _, qty, price in order.items)

def invoice(customer, lines):
    order = Order(customer)
    for sku, qty, price in lines:
        order.add(sku, qty, price)
    return {"customer": customer, "lines": len(order.items), "total": round(subtotal(order) * (1 + TAX), 2)}
