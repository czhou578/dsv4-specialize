from players import Player

def parse(lines):
    rows = (line.split(",") for line in lines if line.strip())
    return [Player(name.strip(), int(w), int(l)) for name, w, l in rows]

def ranking(players):
    """The highest score first; players with equal scores in the order of their names."""
    return sorted(players, key=lambda p: (-p.score, p.name))

def table(lines):
    return [f"{i}. {p.name} {p.score}" for i, p in enumerate(ranking(parse(lines)), 1)]
