"""
Detects invented case details in model-generated text.

The project claims a guardrail against fabricated specifics. This is
the module that makes that claim testable: given the ticket the model
was shown and the text it produced, it returns every concrete claim in
the output that does not appear in the input.

It lives here rather than inside an eval script so the tests and the
eval harness share one implementation. A guardrail with two copies is a
guardrail with two behaviours.

What it catches, and what it does not:

Caught - specifics a model invents when a vague ticket gives it nothing
to work with: version numbers ("8.7"), case or ticket IDs ("CASE-4471"),
dates and timestamps, standalone multi-digit numbers (user counts,
error codes), and named systems or protocols from a fixed list.

Not caught - an invented *cause* stated in ordinary prose ("this is
probably a firewall rule"), an invented department, or a plausible but
unstated assumption. Those need a judge model or a human, and pretending
otherwise would be the same mistake as reporting an accuracy number for
data the model was never built to handle.

So this is a screen, not a proof. An empty finding list means "this
heuristic found nothing", never "nothing was invented" - and the tests
that use it assert on the specific fabrication types above, which it
does catch reliably.

Deliberately string-matching against the input rather than calling a
model: it is fast, deterministic, needs no API key, and therefore runs
in CI on every commit. A judge model would catch more and run never.
"""

import re

# Two or more digits: user counts, error codes, port numbers. Single
# digits are excluded because they appear too often in ordinary prose
# ("1 user", "step 2") to be a useful signal.
NUMBER_PATTERN = re.compile(r"\b\d{2,}\b")

# 8.7, v11.2, 2.1.4
VERSION_PATTERN = re.compile(r"\bv?\d+\.\d+(?:\.\d+)?\b", re.IGNORECASE)

# CASE-4471, ticket #22190, INC0012345, SR 8891
CASE_ID_PATTERN = re.compile(
    r"\b(?:case|ticket|incident|inc|sr|ref)[\s#:_-]*\d+\b", re.IGNORECASE
)

# 2026-09-08, 08/09/2026, 14:00, "last Tuesday" style day names
DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}:\d{2}"
    r"|monday|tuesday|wednesday|thursday|friday|saturday|sunday"
    r"|january|february|march|april|may|june|july|august|september"
    r"|october|november|december)\b",
    re.IGNORECASE
)

KNOWN_SYSTEM_TERMS = [
    "LDAP", "Active Directory", "SAML", "RADIUS", "Kerberos", "OAuth",
    "OIDC", "MFA", "SSO", "VPN", "Okta", "Azure AD", "RSA SecurID",
    "Authentication Manager", "Identity Router", "Cloud Authentication",
    "Tomcat", "Apache", "IIS", "PostgreSQL", "Oracle", "MySQL", "AM"
]


def _normalise(text: str) -> str:
    return (text or "").lower()


def find_invented_details(ticket_text: str, generated_text: str) -> list:
    """
    Return every specific claim present in `generated_text` but absent
    from `ticket_text`.

    Each finding is a dict with `kind` and `value`, so a caller can
    report by category rather than as one undifferentiated list.
    """

    findings = []
    ticket_lower = _normalise(ticket_text)
    generated = generated_text or ""

    def _new(pattern, kind):
        in_ticket = {m.lower() for m in pattern.findall(ticket_text or "")}
        for match in pattern.findall(generated):
            value = match if isinstance(match, str) else match[0]
            if value.lower() not in in_ticket and value.lower() not in ticket_lower:
                findings.append({"kind": kind, "value": value})

    _new(VERSION_PATTERN, "version")
    _new(CASE_ID_PATTERN, "case_id")
    _new(DATE_PATTERN, "date")

    # Numbers last, and only those not already reported as part of a
    # version or date, so "8.7" is not also flagged as a bare number.
    already = {f["value"].lower() for f in findings}
    for number in NUMBER_PATTERN.findall(generated):
        if number in (ticket_text or ""):
            continue
        if any(number in value for value in already):
            continue
        findings.append({"kind": "number", "value": number})

    generated_lower = _normalise(generated)
    for term in KNOWN_SYSTEM_TERMS:
        if term.lower() in generated_lower and term.lower() not in ticket_lower:
            findings.append({"kind": "system", "value": term})

    return findings


def describe_findings(findings: list) -> str:
    """One-line summary for eval output."""

    if not findings:
        return "clean"

    by_kind = {}
    for finding in findings:
        by_kind.setdefault(finding["kind"], []).append(finding["value"])

    return "; ".join(f"{kind}: {sorted(set(values))}" for kind, values in sorted(by_kind.items()))
