from rules import find_indicators


def codes(text, **kw):
    return {i.code for i in find_indicators(text, **kw)}


def test_detects_payment_and_sensitive_info():
    found = codes("Pay a registration fee and send your bank account number and aadhaar today to start work.")
    assert {"upfront_payment", "sensitive_info"} <= found


def test_detects_free_email_and_messaging_app():
    found = codes("Contact recruiter at hr.jobs@gmail.com or on WhatsApp for details about this position.")
    assert {"free_email_contact", "messaging_app"} <= found


def test_corporate_email_not_flagged():
    assert "free_email_contact" not in codes("Send your CV to careers@acme-corp.com for the engineering role.")


def test_clean_posting_has_no_high_severity():
    text = ("We are looking for a software engineer to design backend services. Requirements include a bachelor "
            "degree, three years of Python experience and strong communication. Benefits include health insurance. "
            "Candidates complete a technical interview and a panel discussion before receiving an offer.")
    assert not [i for i in find_indicators(text) if i.severity == "high"]


def test_structural_checks():
    shouty = "URGENT HIRING!!! APPLY NOW!!! EARN MONEY FAST AT HOME WITH NO EXPERIENCE NEEDED WHATSAPP US TODAY"
    found = codes(shouty)
    assert {"excessive_caps", "excessive_exclamation", "very_short", "urgency", "no_experience"} <= found


def test_missing_fields_flagged_only_with_form_fields():
    assert "missing_company_profile" not in codes("some text here", fields=None)
    assert "missing_company_profile" in codes("some text here", fields={"company_profile": "", "requirements": "x"})


def test_sorted_by_severity():
    result = find_indicators("Pay a registration fee. Urgent hiring! Congratulations you are selected.")
    order = [i.severity for i in result]
    assert order == sorted(order, key=["high", "medium", "low"].index)
