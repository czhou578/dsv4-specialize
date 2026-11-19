from board import parse, ranking, table

def test_ties_everywhere():
    lines = ["zed,2,0", "amy,1,0", "kim,2,0", "eve,0,0", "bob,1,0", "al,2,0"]
    assert table(lines) == ["1. al 6", "2. kim 6", "3. zed 6", "4. amy 3", "5. bob 3", "6. eve 0"]

def test_scores_desc_without_ties():
    names = [p.name for p in ranking(parse(["a,0,3", "b,5,0", "c,2,1"]))]
    assert names == ["b", "c", "a"]
