import json
from record import parse_record

def test_examples():
    for line, expected in json.load(open("docs/examples.json")):
        assert parse_record(line) == expected
