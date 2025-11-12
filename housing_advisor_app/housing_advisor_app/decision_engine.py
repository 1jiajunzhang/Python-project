from dataclasses import dataclass
from config import TREND_WEIGHT, RATIO_WEIGHT, AFFORD_WEIGHT

@dataclass
class DecisionInputs:
    trend_score: float
    ratio_score: float
    affordability_score: float

@dataclass
class DecisionResult:
    label: str
    score: float
    explanation: str

class DecisionEngine:
    def __init__(self):
        self.w_trend = TREND_WEIGHT
        self.w_ratio = RATIO_WEIGHT
        self.w_aff  = AFFORD_WEIGHT

    def decide(self, inputs: DecisionInputs) -> DecisionResult:
        s = (
            inputs.trend_score * self.w_trend
            + inputs.ratio_score * self.w_ratio
            + inputs.affordability_score * self.w_aff
        )

        # 阈值
        if s >= 0.38:
            label = "BUY"
            exp = "Trend/valuation/affordability align positively—entry looks attractive."
        elif s >= 0.08:
            label = "WATCH"
            exp = "Mixed signals—keep on watchlist and wait for better entry or seasonal dip."
        else:
            label = "AVOID"
            exp = "Unfavorable risk-weighted mix—pricing/condition not supportive now."

        return DecisionResult(label=label, score=float(round(s, 3)), explanation=exp)
