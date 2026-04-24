def assign_risk_band(pd_score: float) -> str:
    if pd_score < 0.3:
        return "LOW"
    elif pd_score < 0.6:
        return "MEDIUM"
    return "HIGH"

def decision_engine(risk: str) -> str:
    if risk == "LOW":
        return "APPROVE"
    elif risk == "HIGH":
        return "REJECT"
    return "MANUAL REVIEW"
