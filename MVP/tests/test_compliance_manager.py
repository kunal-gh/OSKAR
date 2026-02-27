import pytest

from src.api.compliance_manager import ComplianceManager, RegionMode
from src.models.risk_fusion import RiskFusionEngine


def test_compliance_profiles_exist():
    # Ensure all 3 regions yield valid profiles
    eu_profile = ComplianceManager.get_profile(RegionMode.EU_DSA)
    us_profile = ComplianceManager.get_profile(RegionMode.US_FIRST_AMENDMENT)
    global_profile = ComplianceManager.get_profile(RegionMode.GLOBAL)

    assert eu_profile.hate_threshold == 0.60
    assert us_profile.hate_threshold == 0.95
    assert global_profile.hate_threshold == 0.85

    # EU strictly redacts PII, US does not
    assert eu_profile.strict_pii_redaction is True
    assert us_profile.strict_pii_redaction is False


def test_compliance_routing_divergence():
    """
    Test that the exact same signal values route differently
    based on the loaded compliance profile.
    """
    engine = RiskFusionEngine(num_simulations=10)  # fast mock

    # 1. Subjective Hate Speech + Slight Misinfo without massive bot amplification
    # Under EU DSA (Threshold ~0.60), this is a major violation -> human_review
    # Under US 1A (Threshold ~0.88), this is protected -> soft_warning
    eu_profile = ComplianceManager.get_profile(RegionMode.EU_DSA)
    us_profile = ComplianceManager.get_profile(RegionMode.US_FIRST_AMENDMENT)

    # 0.50 misinfo * 0.6 + 0.95 hate * 0.4 = 0.30 + 0.38 = 0.68 base risk
    # This sits cleanly between the EU (~0.60) and US (~0.88) thresholds.
    eu_res = engine.calculate_risk(
        hate_score=0.95,
        misinfo_score=0.50,
        bot_score=0.10,
        trust_score=0.50,  # neutral
        profile=eu_profile,
    )

    us_res = engine.calculate_risk(
        hate_score=0.95,
        misinfo_score=0.50,
        bot_score=0.10,
        trust_score=0.50,  # neutral
        profile=us_profile,
    )

    # EU should escalate this subjective hate speech to Human Review
    # (because its mean_risk will naturally exceed the low 0.6 HR avg cutoff)
    assert eu_res["route"] in ["human_review", "hard_block"]

    # US should ignore this subjective hate speech and return auto_action or soft_warning
    assert us_res["route"] in ["auto_action", "soft_warning"]


def test_eu_dsa_strict_takedown():
    """
    Test that IF a message is both Hateful AND posted by a Bot Swarm,
    EU DSA mode skips human review and triggers an immediate hard_block.
    """
    engine = RiskFusionEngine(num_simulations=10)
    eu_profile = ComplianceManager.get_profile(RegionMode.EU_DSA)

    res = engine.calculate_risk(
        hate_score=0.99,
        misinfo_score=0.10,
        bot_score=0.99,  # Huge bot swarm signal
        trust_score=0.01,
        profile=eu_profile,
    )

    assert res["route"] == "hard_block"
