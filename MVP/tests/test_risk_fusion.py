from src.models.risk_fusion import RiskFusionEngine


def test_risk_fusion_schema():
    engine = RiskFusionEngine(num_simulations=10)
    result = engine.calculate_risk(
        hate_score=0.9, misinfo_score=0.8, bot_score=0.0, trust_score=0.1
    )

    assert "mean_risk" in result
    assert "confidence_interval" in result
    assert "route" in result
    assert isinstance(result["mean_risk"], float)
    assert len(result["confidence_interval"]) == 2
    assert result["route"] in ["auto_action", "soft_warning", "human_review"]


def test_risk_fusion_logic():
    engine = RiskFusionEngine(num_simulations=100)

    # Trusted user with clean content
    clean_res = engine.calculate_risk(
        hate_score=0.0, misinfo_score=0.0, bot_score=0.0, trust_score=0.9
    )
    assert clean_res["route"] == "auto_action"
    assert clean_res["mean_risk"] < 0.2

    # Untrusted user with terrible content
    toxic_res = engine.calculate_risk(
        hate_score=1.0, misinfo_score=0.9, bot_score=0.9, trust_score=0.1
    )
    assert toxic_res["route"] == "human_review"
    assert toxic_res["mean_risk"] > 0.7
