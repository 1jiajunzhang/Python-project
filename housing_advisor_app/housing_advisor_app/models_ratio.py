def price_to_rent_ratio(price: float, monthly_rent: float) -> float:
    if not price or not monthly_rent or monthly_rent <= 0:
        return 0.0
    return float(price) / float(monthly_rent * 12)


def ratio_score(ptr: float) -> float:
    if ptr <= 0:
        return 0.0
    if ptr < 15:
        return 0.8
    if ptr < 20:
        return 0.4
    if ptr < 25:
        return 0.0
    if ptr < 30:
        return -0.4
    return -0.8


def affordability_score(price: float, income: float, rate: float = 0.06) -> float:
    if not price or not income or income <= 0:
        return 0.0
    annual_payment = price * rate
    dti = annual_payment / income
    if dti < 0.2:
        return 0.8
    if dti < 0.3:
        return 0.4
    if dti < 0.4:
        return 0.0
    if dti < 0.5:
        return -0.4
    return -0.8
