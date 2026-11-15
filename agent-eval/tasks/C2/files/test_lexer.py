import pytest
from lexer import tokenize

def test_basic():
    assert tokenize("1 + 2") == [("NUM", "1"), ("OP", "+"), ("NUM", "2")]
    assert tokenize("x*(y-3.5)") == [("NAME", "x"), ("OP", "*"), ("OP", "("), ("NAME", "y"),
                                     ("OP", "-"), ("NUM", "3.5"), ("OP", ")")]
    with pytest.raises(ValueError):
        tokenize("a ? b")
