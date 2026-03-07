"""Intent gate — decides whether retrieval is needed and classifies intent."""

from __future__ import annotations

import re

# ── Keyword / pattern sets per intent ────────────────────────────

_INTENT_PATTERNS: dict[str, list[str]] = {
    "faq": [
        r"\bfaq\b",
        r"\bfrequently asked\b",
        r"\bhow (does|do|can|to)\b",
        r"\bwhat (is|are|does|do|happens|if)\b",
        r"\bwhy (is|are|does|do)\b",
        r"\bcan i\b",
        r"\bis (it|there|this)\b",
        r"\bwho (is|can|should)\b",
        r"\bwhen (does|do|can|will|is)\b",
        r"\btypes of loan\b",
        r"\bloan (types|products|options)\b",
        r"\bapply (for|online)\b",
        r"\bapplication (status|track|process)\b",
        r"\btrack(ing)?\b",
        r"\bstatus\b",
        r"\beligib(le|ility)\b",
        r"\bapproval (time|process|how long)\b",
        r"\bdisburse(ment|d)?\b",
        r"\bbalance transfer\b",
        r"\bco[- ]?applicant\b",
        r"\bguarantor\b",
        r"\bcustomer (care|support|helpline)\b",
        r"\bgrievance\b",
        r"\bcomplaint\b",
        r"\bhidden charge\b",
        r"\bemi\b",
        r"\btenure\b",
        r"\brepay(ment)?\b",
        r"\bnoc\b",
        r"\bcertificate\b",
        r"\binterest (rate|certificate)\b",
        r"\bcibil\b",
        r"\bcredit score\b",
    ],
    "docs": [
        r"\bdocument(s|ation)?\b",
        r"\bkyc\b",
        r"\bpan\b",
        r"\baadhaar\b",
        r"\bpassport\b",
        r"\bsalary slip\b",
        r"\bitr\b",
        r"\bincome (tax|proof)\b",
        r"\bbank statement\b",
        r"\baddress proof\b",
        r"\bidentity proof\b",
        r"\bproperty document\b",
        r"\bencumbrance\b",
        r"\btitle deed\b",
        r"\brequired (documents|papers|docs)\b",
        r"\bsubmit (documents|docs|papers)\b",
        r"\bwhat (documents|papers|docs)\b",
    ],
    "fees": [
        r"\bfee(s)?\b",
        r"\bcharge(s|d)?\b",
        r"\bprocessing fee\b",
        r"\bprepay(ment)?\b",
        r"\bforeclos(e|ure)\b",
        r"\bpenal(ty|ties|t)?\b",
        r"\blate payment\b",
        r"\bbounce\b",
        r"\bgst\b",
        r"\bstamp(ing)? (duty|charges)\b",
        r"\binsurance (premium|charge|cost)\b",
        r"\bhow much\b.*\b(cost|charge|fee)\b",
        r"\bwaiv(e|er)\b",
        r"\bhidden\b",
        r"\bnach\b.*\b(charge|fee|swap)\b",
    ],
    "policy": [
        r"\bpolic(y|ies)\b",
        r"\bterms (and|&) conditions\b",
        r"\bterms\b",
        r"\brbi\b",
        r"\bregulat(ion|ory|or)\b",
        r"\bprivacy\b",
        r"\bdata (protection|privacy|security|safe)\b",
        r"\bdpdpa\b",
        r"\bconsent\b",
        r"\bsarfaesi\b",
        r"\bnpa\b",
        r"\bdefault\b",
        r"\brecovery\b",
        r"\bombudsman\b",
        r"\bmortgage\b",
        r"\bsecur(ity|ed)\b.*\b(loan|property)\b",
        r"\blien\b",
        r"\bguarantee\b",
        r"\bforce majeure\b",
        r"\bassignment\b",
        r"\bseverability\b",
    ],
    "objection": [
        r"\btoo (high|expensive|much)\b",
        r"\brate (is|seems|looks)\b",
        r"\blower rate\b",
        r"\breduce\b",
        r"\bcan.t (pay|afford)\b",
        r"\bdon.t want\b",
        r"\bnot comfortable\b",
        r"\bwhy (should|do) i\b",
        r"\banother (bank|lender|nbfc)\b",
        r"\bcompetitor\b",
        r"\bworried\b",
        r"\bconcern(ed)?\b",
        r"\bdata (safe|leak|breach)\b",
        r"\breject(ed|ion)\b",
        r"\bnot approved\b",
        r"\bmissed emi\b",
        r"\bcan.t pay\b",
        r"\binsurance.*(don.t|not|no)\b",
    ],
}

# Intents that trigger retrieval
_RETRIEVAL_INTENTS = {"faq", "docs", "fees", "policy", "objection"}


def classify(query: str) -> tuple[str, bool]:
    """Return ``(intent, should_retrieve)`` for the given user query.

    Intent is one of: faq, docs, fees, policy, objection, out_of_scope.
    """
    q = query.lower().strip()
    scores: dict[str, int] = {intent: 0 for intent in _INTENT_PATTERNS}

    for intent, patterns in _INTENT_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, q):
                scores[intent] += 1

    if max(scores.values()) == 0:
        return "out_of_scope", False

    best_intent = max(scores, key=scores.get)  # type: ignore[arg-type]
    should_retrieve = best_intent in _RETRIEVAL_INTENTS
    return best_intent, should_retrieve
