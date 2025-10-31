import logging, random
log = logging.getLogger("simulator")

def clamp_price(p: float) -> float:
    # never allow non-positive prices
    return max(p, 0.01)

class Simulator:
    def __init__(self, start_price=100.0, vol=0.5):
        self.p = start_price
        self.vol = vol

    def next_bar(self):
        drift = random.uniform(-self.vol, self.vol)
        self.p = clamp_price(self.p * (1.0 + drift/100.0))
        return {"open": self.p, "high": self.p*1.001, "low": self.p*0.999, "close": self.p}
