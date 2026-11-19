import pytest
from lexer import tokenize

def test_tokens():
    assert tokenize("2**3*4") == [("NUM", "2"), ("OP", "**"), ("NUM", "3"), ("OP", "*"), ("NUM", "4")]
    assert tokenize("***") == [("OP", "**"), ("OP", "*")]
    assert tokenize("3x") == [("NUM", "3"), ("NAME", "x")]
    assert tokenize("_a1\tb2\n") == [("NAME", "_a1"), ("NAME", "b2")]
    assert tokenize("-007.50") == [("OP", "-"), ("NUM", "007.50")]
    assert tokenize("") == [] and tokenize(" \t\n") == []

@pytest.mark.parametrize("text", ["1.", ".5", "1.2.3", "é", "a=b", "x²"])
def test_errors(text):
    with pytest.raises(ValueError):
        tokenize(text)
