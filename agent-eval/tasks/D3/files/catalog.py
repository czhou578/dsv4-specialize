from memo import invalidate, memo

def discount(qty):
    return 0.9 if qty >= 10 else 1.0

class Catalog:
    def __init__(self, prices):
        self.prices = dict(prices)

    def set_price(self, sku, value):
        self.prices[sku] = value
        invalidate(self)

    @memo
    def unit(self, sku):
        return self.prices[sku]

    @memo
    def quote(self, sku, qty):
        return round(self.unit(sku) * qty * discount(qty), 2)
