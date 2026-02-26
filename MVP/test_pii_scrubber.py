"""
test_pii_scrubber.py — OSKAR v0.6
Tests PIIScrubber: email, SSN, phone, and credit card redaction 
before text hits the NLP pipelines.
"""

import pytest
from pii_scrubber import PIIScrubber


def test_pii_scrubber_initialization():
    scrubber = PIIScrubber(mode="redact")
    assert scrubber.mode == "redact"


def test_no_pii_returns_clean_text():
    scrubber = PIIScrubber()
    original = "This is a clean post about climate change policy."
    res = scrubber.scrub(original)
    
    assert res["clean_text"] == original
    assert res["pii_found"] is False
    assert len(res["redactions"]) == 0


def test_email_redaction():
    scrubber = PIIScrubber()
    original = "Contact me at alice.smith@example.com for more info."
    res = scrubber.scrub(original)
    
    assert "alice.smith@example.com" not in res["clean_text"]
    assert "<EMAIL>" in res["clean_text"]
    assert res["pii_found"] is True
    assert "email" in res["redactions"]


def test_ssn_redaction():
    scrubber = PIIScrubber()
    original = "My fake SSN is 123-45-6789 do not share it."
    res = scrubber.scrub(original)
    
    assert "123-45-6789" not in res["clean_text"]
    assert "<SSN>" in res["clean_text"]
    assert "ssn" in res["redactions"]


def test_credit_card_redaction():
    scrubber = PIIScrubber()
    original = "I think the number is 4111 1111 2222 3333 right?"
    res = scrubber.scrub(original)
    
    # Needs to catch 16 digit sequences with spaces
    assert "4111 1111 2222 3333" not in res["clean_text"]
    assert "CREDIT_CARD" in res["clean_text"]
    assert "credit_card" in res["redactions"]


def test_phone_number_redaction():
    scrubber = PIIScrubber()
    original = "Call the office at (555) 123-4567 before noon."
    res = scrubber.scrub(original)
    
    assert "(555) 123-4567" not in res["clean_text"]
    assert "PHONE" in res["clean_text"]
    assert "phone" in res["redactions"]


def test_multiple_pii_redaction():
    """Verify that multiple stacked types of PII are all caught in one pass."""
    scrubber = PIIScrubber(mode="mask") # Test masking mode
    original = "Email bob@test.co or call 800-555-0199. CC: 1234-5678-9012-3456"
    res = scrubber.scrub(original)
    
    assert "bob@test.co" not in res["clean_text"]
    assert "800-555-0199" not in res["clean_text"]
    assert "1234-5678-9012-3456" not in res["clean_text"]
    
    # Mask mode replaces with asterisks proportional to length
    assert "***********" in res["clean_text"]
    assert "************" in res["clean_text"]
    assert "*******************" in res["clean_text"]
    
    assert res["pii_found"] is True
    assert set(res["redactions"]) == {"email", "phone", "credit_card"}
