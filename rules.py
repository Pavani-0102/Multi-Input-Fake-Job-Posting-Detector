"""Transparent, rule-based suspicious-indicator checks.

These rules are independent of the ML model: each one is a visible regular expression
or simple statistic, and the matched text is shown to the user. They are heuristics
that highlight common scam patterns, not evidence of fraud.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Optional

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

FREE_EMAIL_DOMAINS = (
    "gmail", "yahoo", "outlook", "hotmail", "live", "rediffmail", "aol",
    "protonmail", "icloud", "mail", "gmx", "yandex",
)


@dataclass(frozen=True)
class Indicator:
    code: str
    title: str
    detail: str
    severity: str  # "high" | "medium" | "low"


@dataclass(frozen=True)
class _PatternRule:
    code: str
    title: str
    severity: str
    explanation: str
    pattern: re.Pattern


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


_PATTERN_RULES: tuple[_PatternRule, ...] = (
    _PatternRule(
        "upfront_payment", "Requests payment or a fee", "high",
        "Legitimate employers do not normally ask candidates to pay registration, "
        "training, security-deposit or processing fees.",
        _rx(
            r"\b(registration|training|security|processing|application|joining|refundable|"
            r"verification|kit|onboarding)\s+(fee|fees|deposit|charges?|amount|payment)\b"
            r"|\bpay\s+(a\s+|the\s+)?(small\s+)?(fee|deposit|amount)\b"
            r"|\b(send|transfer|deposit)\s+(the\s+)?(money|payment|amount|fee)\b"
        ),
    ),
    _PatternRule(
        "sensitive_info", "Asks for sensitive personal or financial details", "high",
        "Requests for bank, identity or card details at the application stage are a "
        "common phishing pattern.",
        _rx(
            r"\b(bank\s+account|account\s+number|aadhaar|aadhar|pan\s+card|ssn|"
            r"social\s+security|passport\s+(copy|scan|number)|credit\s+card|debit\s+card|"
            r"cvv|otp|ifsc|routing\s+number)\b"
        ),
    ),
    _PatternRule(
        "unrealistic_pay", "Unrealistic or guaranteed earnings", "high",
        "Very high or guaranteed pay for minimal work is a frequent scam hook.",
        _rx(
            r"\bguaranteed\s+(income|salary|earnings|job|placement)\b"
            r"|\beasy\s+money\b|\bget\s+rich\b|\bearn\s+(up\s+to\s+)?(\$|usd|rs\.?|inr|₹|£|€)?\s?\d[\d,]{2,}\s*"
            r"(per|/|a)\s*(day|daily|week|weekly)\b"
            r"|\b(\$|₹|rs\.?)\s?\d[\d,]{3,}\s*(per|/|a)\s*(day|week)\b"
            r"|\bearn\s+from\s+home\b|\bdaily\s+payout\b|\bweekly\s+payout\b"
        ),
    ),
    _PatternRule(
        "no_interview", "No interview or screening mentioned", "medium",
        "Offers that skip interviews or screening are unusual for genuine roles.",
        _rx(
            r"\bno\s+interview\b|\bwithout\s+(an\s+)?interview\b|\bdirect\s+(joining|selection)\b"
            r"|\binstant\s+(joining|selection|offer)\b|\bimmediate\s+joining\b"
            r"|\bselected\s+without\b"
        ),
    ),
    _PatternRule(
        "no_experience", "No experience or qualifications needed", "medium",
        "Claims that anyone can do the job, often combined with high pay.",
        _rx(
            r"\bno\s+(prior\s+)?(experience|qualification|skills?|degree)\s+(is\s+)?"
            r"(needed|required|necessary)\b|\bno\s+experience\b|\banyone\s+can\s+(apply|do|earn)\b"
        ),
    ),
    _PatternRule(
        "urgency", "Pressure or urgency language", "medium",
        "Scams often rush candidates so they act before verifying.",
        _rx(
            r"\burgent(ly)?\s+(hiring|required|requirement|need)\b|\blimited\s+(seats|slots|vacancies|positions)\b"
            r"|\bapply\s+(immediately|now|today)\b|\bhurry\b|\bact\s+now\b|\btoday\s+only\b"
            r"|\blast\s+(date|chance)\b|\bonly\s+\d+\s+(seats|slots|positions|vacancies)\b"
        ),
    ),
    _PatternRule(
        "messaging_app", "Contact through WhatsApp / Telegram", "medium",
        "Recruiters using personal messaging apps instead of official channels warrant caution.",
        _rx(r"\b(whatsapp|telegram|signal|wechat)\b"),
    ),
    _PatternRule(
        "shortened_link", "Shortened or obscured link", "medium",
        "Shortened URLs hide the real destination.",
        _rx(r"\b(bit\.ly|tinyurl\.com|t\.co|goo\.gl|cutt\.ly|rb\.gy|is\.gd|shorturl\.at)/\S*"),
    ),
    _PatternRule(
        "unsolicited_selection", "Claims you were already selected", "low",
        "Genuine employers rarely announce selection before any application or interview.",
        _rx(r"\bcongratulations\b|\byou\s+(have\s+been|are|were)\s+(selected|shortlisted|chosen)\b"),
    ),
    _PatternRule(
        "work_from_home_data_entry", "Generic work-from-home / data-entry task", "low",
        "Vague remote 'typing', 'data entry' or 'form filling' jobs are commonly used in scams.",
        _rx(r"\b(copy\s*paste|typing\s+jobs?|form\s+filling|ad\s+posting|like\s+and\s+(share|subscribe))\b"),
    ),
)

_EMAIL_FINDER = re.compile(r"\b[\w.+-]+@([\w-]+)(?:\.[\w-]+)+\b")


def _snippet(text: str, match: re.Match, width: int = 40) -> str:
    start = max(match.start() - width, 0)
    end = min(match.end() + width, len(text))
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{snippet}{suffix}"


def find_indicators(text: str, fields: Optional[Mapping[str, str]] = None) -> list[Indicator]:
    """Run all transparent checks and return the indicators that fired.

    Parameters
    ----------
    text:   the full text being analysed.
    fields: optional structured form fields (Mode A) enabling missing-field checks.
    """
    indicators: list[Indicator] = []
    text = text or ""

    for rule in _PATTERN_RULES:
        match = rule.pattern.search(text)
        if match:
            indicators.append(
                Indicator(rule.code, rule.title, f"{rule.explanation} Matched: \"{_snippet(text, match)}\"", rule.severity)
            )

    for m in _EMAIL_FINDER.finditer(text):
        domain = m.group(1).lower()
        if domain in FREE_EMAIL_DOMAINS:
            indicators.append(
                Indicator(
                    "free_email_contact",
                    "Contact uses a free personal e-mail domain",
                    f"Company recruiters normally use a corporate domain. Matched: \"{m.group(0)}\"",
                    "medium",
                )
            )
            break

    letters = [c for c in text if c.isalpha()]
    if len(letters) >= 50:
        upper_ratio = sum(c.isupper() for c in letters) / len(letters)
        if upper_ratio > 0.40:
            indicators.append(
                Indicator("excessive_caps", "Excessive capital letters", f"{upper_ratio:.0%} of letters are uppercase.", "low")
            )

    if text.count("!") >= 3:
        indicators.append(
            Indicator("excessive_exclamation", "Excessive exclamation marks", f"{text.count('!')} exclamation marks found.", "low")
        )

    word_count = len(text.split())
    if 0 < word_count < 40:
        indicators.append(
            Indicator("very_short", "Very short or vague posting", f"Only {word_count} words; little concrete detail on duties or company.", "low")
        )

    if fields is not None:
        if not (fields.get("company_profile") or "").strip():
            indicators.append(
                Indicator("missing_company_profile", "No company profile provided", "Postings without any company information are harder to verify.", "low")
            )
        if not (fields.get("requirements") or "").strip():
            indicators.append(
                Indicator("missing_requirements", "No job requirements provided", "Genuine roles usually list concrete requirements.", "low")
            )

    indicators.sort(key=lambda i: (SEVERITY_ORDER[i.severity], i.code))
    return indicators
