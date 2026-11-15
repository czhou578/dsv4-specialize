from dataclasses import dataclass

@dataclass
class Player:
    name: str
    wins: int
    losses: int

    @property
    def score(self):
        return 3 * self.wins - self.losses
