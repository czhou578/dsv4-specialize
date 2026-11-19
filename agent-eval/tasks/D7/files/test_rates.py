import json
from pathlib import Path

from rates import convert, load_rates

FIX = Path(__file__).parent / "fixtures"

def test_load():
    assert load_rates(FIX / "rates.csv")["EUR"] == 0.92

def test_convert_golden():
    rates = load_rates(FIX / "rates.csv")
    for case in json.loads((FIX / "golden.json").read_text()):
        assert convert(case["amount"], case["from"], case["to"], rates) == case["expected"]
