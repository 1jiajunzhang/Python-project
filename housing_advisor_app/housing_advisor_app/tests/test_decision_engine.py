from decision_engine import DecisionEngine, DecisionInputs


def test_decision_engine_basic():
    engine = DecisionEngine()
    hi = DecisionInputs(0.8, 0.8, 0.8, 0.5)
    lo = DecisionInputs(-0.5, -0.5, -0.5, -0.5)
    r_hi = engine.decide(hi)
    r_lo = engine.decide(lo)
    assert r_hi.label in ("BUY", "WATCH")
    assert r_lo.label in ("AVOID", "WATCH")
