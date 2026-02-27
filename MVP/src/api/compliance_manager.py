from enum import Enum

from pydantic import BaseModel


class RegionMode(str, Enum):
    GLOBAL = "GLOBAL"
    EU_DSA = "EU_DSA"
    US_FIRST_AMENDMENT = "US_FIRST_AMENDMENT"


class ComplianceProfile(BaseModel):
    hate_threshold: float
    bot_threshold: float
    misinfo_threshold: float
    requires_human_verification: bool
    strict_pii_redaction: bool


class ComplianceManager:
    """
    Dynamically adjusts OSKAR routing thresholds based on regional legislation.
    """

    @staticmethod
    def get_profile(region: RegionMode) -> ComplianceProfile:
        if region == RegionMode.EU_DSA:
            # EU Digital Services Act:
            # - Extremely low tolerance for Hate Speech and Misinformation
            # - High emphasis on Human Verification
            # - Strict PII Data Privacy (GDPR)
            return ComplianceProfile(
                hate_threshold=0.60,
                bot_threshold=0.60,
                misinfo_threshold=0.60,
                requires_human_verification=True,
                strict_pii_redaction=True,
            )

        elif region == RegionMode.US_FIRST_AMENDMENT:
            # US 1st Amendment / Section 230:
            # - High tolerance for subjective text (Misinfo/Hate)
            # - Focuses enforcement on automated behavior (Bot Swarms)
            return ComplianceProfile(
                hate_threshold=0.95,
                bot_threshold=0.75,
                misinfo_threshold=0.95,
                requires_human_verification=False,
                strict_pii_redaction=False,
            )

        else:
            # Standard / Default / Global baseline
            return ComplianceProfile(
                hate_threshold=0.85,
                bot_threshold=0.85,
                misinfo_threshold=0.85,
                requires_human_verification=False,
                strict_pii_redaction=True,
            )
