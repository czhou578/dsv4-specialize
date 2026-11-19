import re

TOKEN = re.compile(r"(?P<NUM>[0-9]+(\.[0-9]+)?)|(?P<NAME>[A-Za-z_][A-Za-z0-9_]*)|(?P<OP>\*\*|[-+*/()])|(?P<WS>[ \t\n]+)")

def tokenize(text):
    out, pos = [], 0
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f"bad character at {pos}")
        if m.lastgroup != "WS":
            out.append((m.lastgroup, m.group()))
        pos = m.end()
    return out
