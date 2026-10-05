from __future__ import annotations

SAFE_MAX       = 35   # 0  – 35  → Safe
SUSPICIOUS_MAX = 69   # 36 – 69  → Suspicious
                      # 70 – 100 → Unsafe


def _verdict(score: float) -> str:
    if score <= SAFE_MAX:
        return "safe"
    if score <= SUSPICIOUS_MAX:
        return "suspicious"
    return "unsafe"


def fuse(p_dl: float, p_ml: float | None) -> dict:
    """
    Combine DL and ML phishing probabilities into a single risk score.

    confidence = |p - 0.5|  (0 = no idea, 0.5 = maximally certain)
    final_prob = (p_dl * conf_dl + p_ml * conf_ml) / (conf_dl + conf_ml)

    Edge cases:
      - p_ml is None  → partial analysis, use p_dl directly.
      - both confs == 0 → simple average (both equally uncertain).
    """
    p_dl    = float(p_dl)
    conf_dl = abs(p_dl - 0.5)

    # ── Partial analysis (ML unavailable) ─────────────────────────────────────
    if p_ml is None:
        risk_score = p_dl * 100.0
        return {
            "risk_score":     round(risk_score, 2),
            "verdict":        _verdict(risk_score),
            "dl_probability": round(p_dl, 4),
            "ml_probability": None,
            "dl_confidence":  round(conf_dl * 2, 4),
            "ml_confidence":  None,
            "analysis_type":  "partial",
        }

    # ── Full analysis ──────────────────────────────────────────────────────────
    p_ml    = float(p_ml)
    conf_ml = abs(p_ml - 0.5)

    total_conf = conf_dl + conf_ml

    if total_conf == 0.0:
        final_prob = (p_dl + p_ml) / 2.0
    else:
        final_prob = (p_dl * conf_dl + p_ml * conf_ml) / total_conf

    risk_score = final_prob * 100.0

    return {
        "risk_score":     round(risk_score, 2),
        "verdict":        _verdict(risk_score),
        "dl_probability": round(p_dl,  4),
        "ml_probability": round(p_ml,  4),
        "dl_confidence":  round(conf_dl * 2, 4),
        "ml_confidence":  round(conf_ml * 2, 4),
        "analysis_type":  "full",
    }
