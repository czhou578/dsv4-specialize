import csv

def load_rates(path):
    """Currency -> units per 1 USD, from a CSV file with the columns currency,rate."""
    with open(path, newline="") as f:
        return {row["currency"]: float(row["rate"]) for row in csv.DictReader(f)}

def convert(amount, src, dst, rates):
    usd = amount if src == "USD" else amount / rates[src]
    return round(usd if dst == "USD" else usd * rates[dst], 2)
