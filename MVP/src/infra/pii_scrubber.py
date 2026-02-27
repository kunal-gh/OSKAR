"""
pii_scrubber.py — OSKAR v0.6 Enterprise Security
------------------------------------------------
Scans and redacts Personally Identifiable Information (PII)
from text before it reaches the NLP analysis pipeline.

Supported RegEx Redactions:
- Email addresses
- Phone numbers
- Social Security Numbers (US)
- Credit Card Numbers

Why? To ensure OSKAR is safe to use in enterprise environments
where sensitive user data might accidentally be pasted into
chat platforms or review queues.
"""

import re

# Regex Patterns for common PII
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Matches various common US phone formats: (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
PHONE_REGEX = re.compile(r"\+?1?\s*\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")

# Standard 9-digit US SSN (XXX-XX-XXXX or XXXXXXXXX)
SSN_REGEX = re.compile(
    r"\b(?!(000|666|9))\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b|\b(?!(000|666|9))\d{9}\b"
)

# Basic 16-digit credit card math (Visa/Mastercard)
CREDIT_CARD_REGEX = re.compile(r"\b(?:\d[ -]*?){13,16}\b")


class PIIScrubber:
    """
    Sanitizes user input by redacting raw PII before NLP processing.
    """

    def __init__(self, mode="redact"):
        # "redact" = Replace with <PII_TYPE>
        # "mask"   = Replace with ***
        self.mode = mode

    def _replace(self, match, token_name: str) -> str:
        if self.mode == "mask":
            return "*" * len(match.group(0))
        return f"<{token_name}>"

    def scrub(self, text: str) -> dict:
        """
        Scans text for PII, returns the scrubbed text and a flag if PII was found.
        """
        if not text:
            return {"clean_text": text, "pii_found": False, "redactions": []}

        original_text = text
        redactions = []

        # 1. Scrape Emails
        if EMAIL_REGEX.search(text):
            text = EMAIL_REGEX.sub(lambda m: self._replace(m, "EMAIL"), text)
            redactions.append("email")

        # 2. Scrape SSNs
        if SSN_REGEX.search(text):
            text = SSN_REGEX.sub(lambda m: self._replace(m, "SSN"), text)
            redactions.append("ssn")

        # 3. Scrape Credit Cards (Very broad, runs after SSN to avoid overlap)
        # Regex checks for 13-16 digits with optional spaces/dashes.
        if CREDIT_CARD_REGEX.search(text):
            # Verify it's actually just digits/spaces/dashes and long enough
            # (Basic check to avoid redacting random long numbers, though a real system uses Luhn's)
            potential_ccs = CREDIT_CARD_REGEX.findall(text)
            for cc in potential_ccs:
                digits = re.sub(r"\D", "", cc)
                if 13 <= len(digits) <= 16:
                    text = text.replace(cc, self._replace(re.match(r".*", cc), "CREDIT_CARD"))
                    if "credit_card" not in redactions:
                        redactions.append("credit_card")

        # 4. Scrape Phone Numbers
        if PHONE_REGEX.search(text):
            # The phone regex is extremely broad, we do a secondary check to ensure it looks like a US number
            # and isn't just a randomly matched date or something similar if possible.
            # But for v0.6 MVP, we trust the regex.
            text = PHONE_REGEX.sub(lambda m: self._replace(m, "PHONE"), text)
            if "phone" not in redactions:
                redactions.append("phone")

        return {"clean_text": text, "pii_found": len(redactions) > 0, "redactions": redactions}
