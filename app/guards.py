"""
Prompt injection detection and scope guard.
These checks run BEFORE the LLM to prevent misuse.
"""

import re
import logging

log = logging.getLogger(__name__)

# ─── Prompt Injection Patterns ────────────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(\w+\s+)*(previous|all|above|prior|instructions)",
    r"disregard\s+(\w+\s+)*(previous|all|above|prior|instructions)",
    r"you\s+are\s+now\s+(a|an|the)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a|an|the)\s+",
    r"new\s+(instructions?|directives?|rules?|role):",
    r"system\s+prompt",
    r"jailbreak",
    r"\bDAN\b",
    r"developer\s+mode",
    r"unrestricted\s+mode",
    r"bypass\s+(safety|filter|restriction|guard)",
    r"forget\s+(your\s+)?(previous\s+)?(instructions?|training|rules?)",
    r"override\s+(previous\s+)?(instructions?|rules?|constraints?)",
    r"<\s*/?system\s*>",
    r"<\s*/?human\s*>",
    r"<\s*/?assistant\s*>",
    r"###\s*(instruction|system|override)",
    r"reveal\s+(your\s+)?(system\s+)?(prompt|instructions?)",
    r"print\s+(your\s+)?(system\s+)?(prompt|instructions?)",
    r"what\s+(are\s+)?(your\s+)?(exact\s+)?(system\s+)?(prompt|instructions?)",
]

_INJECTION_REGEX = re.compile(
    "|".join(INJECTION_PATTERNS),
    re.IGNORECASE | re.DOTALL
)

# ─── Off-Topic Categories ─────────────────────────────────────────────────────
OFF_TOPIC_PATTERNS = [
    # Legal / compliance
    r"\b(lawsuit|litigation|gdpr|hipaa|ada\s+compliance|discrimination|wrongful\s+(termination|dismissal))\b",
    # General HR (not assessment-related)
    r"\b(salary|compensation|pay\s+(scale|grade)|benefits|vacation|pto|sick\s+leave)\b",
    r"\b(fire|layoff|redundanc|terminate\s+(employee|staff))\b",
    r"\b(interview\s+(question|technique|tip)|how\s+to\s+interview)\b",
    # Competitor products (refuse to recommend competitors)
    r"\b(hackerrank|codility|pymetrics|criteria\s+corp|hirevu|predictive\s+index|wonderlic|talentplus)\b",
    # Personal / sensitive
    r"\b(my\s+(password|account|login|credit\s+card))\b",
    # General chatting
    r"\b(tell\s+me\s+a\s+joke|what\s+is\s+the\s+weather|write\s+(me\s+)?(a\s+)?(poem|story|essay))\b",
]

_OFF_TOPIC_REGEX = re.compile(
    "|".join(OFF_TOPIC_PATTERNS),
    re.IGNORECASE
)

# ─── SHL-relevant keywords (positive signal) ──────────────────────────────────
SHL_KEYWORDS = [
    "assessment", "test", "evaluate", "evaluation", "hire", "hiring", "recruit",
    "candidate", "role", "job", "position", "developer", "manager", "analyst",
    "shl", "opq", "verify", "personality", "cognitive", "ability", "aptitude",
    "simulation", "knowledge", "skills", "behavior", "behaviour", "java", "python",
    "sales", "customer service", "leadership", "graduate", "entry level", "senior",
    "mid-level", "executive", "compare", "difference", "recommend", "suggest",
    "shortlist", "what assessments", "which test", "help me find", "looking for",
]


def is_injection_attempt(text: str) -> bool:
    """Return True if text appears to be a prompt injection attempt."""
    return bool(_INJECTION_REGEX.search(text))


def is_off_topic(text: str) -> bool:
    """
    Return True if text is clearly off-topic (not related to SHL assessments).
    Uses a combination of off-topic patterns + absence of SHL-relevant keywords.
    Conservative: only flag if clearly off-topic AND no SHL keywords present.
    """
    has_off_topic = bool(_OFF_TOPIC_REGEX.search(text))
    if not has_off_topic:
        return False

    # If it also mentions SHL-relevant terms, don't flag (e.g., "salary for Java developer")
    text_lower = text.lower()
    has_shl_keywords = any(kw in text_lower for kw in SHL_KEYWORDS)
    return not has_shl_keywords


def check_message(text: str) -> tuple[bool, str]:
    """
    Check a message for injection or off-topic content.
    Returns (is_safe, reason_if_unsafe).
    """
    if is_injection_attempt(text):
        log.warning(f"Injection attempt detected: {text[:100]}")
        return False, "injection"

    if is_off_topic(text):
        log.info(f"Off-topic message: {text[:100]}")
        return False, "off_topic"

    return True, ""


INJECTION_REPLY = (
    "I'm only able to help with SHL assessment recommendations. "
    "I can't process that type of request."
)

OFF_TOPIC_REPLY = (
    "I specialize in recommending SHL talent assessments. "
    "I'm not able to help with that topic, but I'd be happy to help you find "
    "the right SHL assessments for your hiring needs. "
    "What role are you hiring for?"
)
