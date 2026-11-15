from board import table

def test_table():
    lines = ["cara,4,1", "bo,3,0", "abe,3,0", "dan,1,5"]
    assert table(lines) == ["1. cara 11", "2. abe 9", "3. bo 9", "4. dan -2"]
