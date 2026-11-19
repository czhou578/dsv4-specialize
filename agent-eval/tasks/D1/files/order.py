"""An order of the shop: the customer and a list of (sku, qty, unit price) lines."""

class Order:
    def __init__(self, customer, items=[]):
        self.customer = customer
        self.items = items

    def add(self, sku, qty, price):
        self.items.append((sku, qty, price))
        return self
