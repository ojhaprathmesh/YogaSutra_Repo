"""
app/security/guardrails.py — Input Security & Safety Guardrail Engine

Implements three layers of input protection before any user text reaches
the LLM multi-agent pipeline:

  Layer 1 — Prompt Injection & Jailbreak Detection
      Catches system-override attempts, role-hijacking, delimiter collision,
      "ignore previous instructions" patterns, and indirect injection.

  Layer 2 — Toxicity & Abuse Filtering
      Flags hate speech, harassment, self-harm incitement, profanity-dense
      inputs, and requests for harmful content.

  Layer 3 — Domain Relevance Shield
      Soft-rejects clearly out-of-scope requests (e.g. "write malware",
      "book a flight") and enforces the yoga-health domain boundary.

Usage::

    from app.security.guardrails import run_input_guardrails, GuardrailResult

    result: GuardrailResult = run_input_guardrails(user_input)
    if not result.is_safe:
        raise HTTPException(status_code=400, detail=result.rejection_reason)
    # proceed with result.sanitized_input

Architecture note (PLAN.md §8):
  Guardrail checks run BEFORE InMemoryRunner is invoked. A
  SafetyEscalationEvent is published to the Event Bus on each violation
  for auditability and longitudinal monitoring.
"""
from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

logger = logging.getLogger("yogasutra.security.guardrails")


# ---------------------------------------------------------------------------
# Violation taxonomy
# ---------------------------------------------------------------------------

class ViolationType(str, Enum):
    PROMPT_INJECTION        = "prompt_injection"
    JAILBREAK               = "jailbreak"
    ROLE_HIJACKING          = "role_hijacking"
    DELIMITER_COLLISION     = "delimiter_collision"
    TOXICITY                = "toxicity"
    HATE_SPEECH             = "hate_speech"
    SELF_HARM               = "self_harm"
    OUT_OF_DOMAIN           = "out_of_domain"
    EXCESSIVE_LENGTH        = "excessive_length"


# ---------------------------------------------------------------------------
# Guardrail result schema (Pydantic-compatible dataclass)
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Structured result of the guardrail inspection pipeline."""
    is_safe: bool
    risk_score: float                        # 0.0 (safe) → 1.0 (critical)
    violations: list[ViolationType] = field(default_factory=list)
    rejection_reason: str = ""
    sanitized_input: str = ""               # cleaned text (only valid when is_safe=True)
    details: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_safe": self.is_safe,
            "risk_score": self.risk_score,
            "violations": [v.value for v in self.violations],
            "rejection_reason": self.rejection_reason,
            "sanitized_input": self.sanitized_input,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Layer 1: Prompt Injection & Jailbreak Patterns
# ---------------------------------------------------------------------------

# Patterns ordered roughly by severity (checked first → break on match)
_INJECTION_PATTERNS: list[tuple[re.Pattern, ViolationType, str]] = [
    # Classic "ignore instructions" attacks
    (re.compile(
        r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|directives?|prompt)",
        re.I
    ), ViolationType.PROMPT_INJECTION, "ignore_previous_instructions"),

    # System-prompt extraction
    (re.compile(
        r"(print|repeat|tell me|reveal|show|output|dump|display)\s+.{0,30}"
        r"(your\s+)?(system\s+prompt|instructions?|initial\s+prompt|base\s+prompt)",
        re.I
    ), ViolationType.PROMPT_INJECTION, "system_prompt_extraction"),

    # Role hijacking: "you are now DAN / an unrestricted AI…"
    (re.compile(
        r"(you\s+are\s+now|act\s+as|pretend\s+(to\s+be|you\s+are)|roleplay\s+as|"
        r"simulate|impersonate|from\s+now\s+on\s+you\s+are)\s+.{0,60}"
        r"(ai|assistant|chatbot|dan|evil|unrestricted|unfiltered|jailbreak)",
        re.I
    ), ViolationType.ROLE_HIJACKING, "role_hijacking"),

    # Jailbreak keywords
    (re.compile(
        r"\b(jailbreak|dan\s+mode|developer\s+mode|god\s+mode|"
        r"bypass\s+(filter|restriction|safety|guardrail)|"
        r"disable\s+(safety|filter|guardrail|restriction))\b",
        re.I
    ), ViolationType.JAILBREAK, "jailbreak_keyword"),

    # Delimiter / markdown injection trying to break prompt structure
    (re.compile(
        r"(<\s*system\s*>|<\s*/\s*system\s*>|<\s*user\s*>|<\s*assistant\s*>|"
        r"\[INST\]|\[\/INST\]|<<SYS>>|<</SYS>>|\|system\||"
        r"###\s*(instruction|system|prompt))",
        re.I
    ), ViolationType.DELIMITER_COLLISION, "delimiter_injection"),

    # Indirect injection: "the following is a message from your developer"
    (re.compile(
        r"(message\s+from\s+(your\s+)?(developer|creator|openai|google|anthropic)|"
        r"as\s+(the\s+)?(creator|developer|admin|root\s+user)\s+I\s+(command|instruct|tell))",
        re.I
    ), ViolationType.PROMPT_INJECTION, "indirect_authority_injection"),
]

# ---------------------------------------------------------------------------
# Layer 2: Toxicity & Abuse Patterns
# ---------------------------------------------------------------------------

_TOXICITY_PATTERNS: list[tuple[re.Pattern, ViolationType, str]] = [
    # Hate speech
    (re.compile(
        r"\b(kill\s+(all|every)\s+\w+|exterminate|genocide|ethnic\s+cleansing|"
        r"racial\s+slur|n[i1]gg[e3]r|k[i1]k[e3]|ch[i1]nk|sp[i1]c)\b",
        re.I
    ), ViolationType.HATE_SPEECH, "hate_speech"),

    # Self-harm & suicide incitement
    (re.compile(
        r"(commit\s+suicide|kill\s+myself|end\s+my\s+life|self[.\-]harm|"
        r"how\s+do\s+i\s+(commit\s+suicide|kill\s+myself)|"
        r"best\s+way\s+to\s+(die|end\s+my\s+life)|"
        r"want\s+to\s+(die|end\s+it\s+all|kill\s+myself)|"
        r"methods?\s+of\s+suicide)",
        re.I
    ), ViolationType.SELF_HARM, "self_harm_incitement"),

    # Explicit harassment / threats
    (re.compile(
        r"\b(i\s+will\s+kill|i('m|\s+am)\s+going\s+to\s+(kill|hurt|destroy)\s+(you|them)|"
        r"you\s+deserve\s+to\s+die|go\s+kill\s+yourself)\b",
        re.I
    ), ViolationType.TOXICITY, "explicit_threat"),
]

# ---------------------------------------------------------------------------
# Layer 3: Out-of-Domain Shield
# ---------------------------------------------------------------------------

_OUT_OF_DOMAIN_PATTERNS: list[re.Pattern] = [
    # Clearly unrelated tech / security requests
    re.compile(
        r"\b(write\s+(me\s+)?(malware|ransomware|virus|exploit|payload|rootkit)|"
        r"sql\s+injection\s+(attack|payload)|penetration\s+test\s+(target|victim)|"
        r"hack\s+(into|this\s+website|their\s+server|a\s+computer))\b",
        re.I,
    ),
    # Financial / booking requests (completely off-domain)
    re.compile(
        r"\b(book\s+(a\s+)?(flight|hotel|ticket|reservation)|"
        r"stock\s+(trading|market\s+advice|pick)|"
        r"write\s+(my\s+)?(essay|thesis|dissertation|code\s+for\s+my\s+homework))\b",
        re.I,
    ),
]

# Broad yoga/wellness topic – used to suppress false positives for borderline
# out-of-domain checks (if the input also mentions these it's likely on-domain).
_YOGA_DOMAIN_SIGNALS = re.compile(
    r"\b(yoga|asana|pranayama|meditation|breathe|chakra|pose|practice|"
    r"flexibility|stress|relax|strength|balance|mindful|savasana|namaste|"
    r"body|muscle|joint|spine|back|shoulder|knee|hip|neck|sleep|anxiety|"
    r"energy|morning|evening|breath|inhale|exhale|stretch|calm|wellness|"
    r"beginner|intermediate|advanced|minutes?|session)\b",
    re.I,
)

# ---------------------------------------------------------------------------
# Input sanitisation helpers
# ---------------------------------------------------------------------------

_MAX_INPUT_CHARS = 2000  # hard cap – longer inputs are suspicious


def _normalize_unicode(text: str) -> str:
    """NFKC-normalise and strip invisible control characters."""
    text = unicodedata.normalize("NFKC", text)
    # Strip control chars (category 'C') except normal whitespace
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C"
        or ch in ("\n", "\r", "\t")
    )


def _strip_suspicious_tags(text: str) -> str:
    """Remove HTML/XML-like tags that could be used for delimiter injection."""
    return re.sub(r"<[^>]{0,80}>", "", text)


def _sanitize(text: str) -> str:
    text = _normalize_unicode(text)
    text = _strip_suspicious_tags(text)
    # Collapse excessive whitespace
    text = re.sub(r"[ \t]{3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Public guardrail runner
# ---------------------------------------------------------------------------

def run_input_guardrails(user_input: str) -> GuardrailResult:
    """
    Run all three guardrail layers against *user_input*.

    Returns a :class:`GuardrailResult`. The caller is responsible for
    rejecting the request when ``result.is_safe is False``.
    """
    violations: list[ViolationType] = []
    details: dict[str, str] = {}
    risk_score = 0.0

    # ── Length check ──────────────────────────────────────────────────────
    if len(user_input) > _MAX_INPUT_CHARS:
        violations.append(ViolationType.EXCESSIVE_LENGTH)
        details["excessive_length"] = (
            f"Input length {len(user_input)} exceeds maximum {_MAX_INPUT_CHARS} characters."
        )
        risk_score = max(risk_score, 0.4)
        # Truncate before further checks
        user_input = user_input[:_MAX_INPUT_CHARS]

    # ── Sanitise raw input ────────────────────────────────────────────────
    sanitized = _sanitize(user_input)

    # ── Layer 1: Injection / Jailbreak ────────────────────────────────────
    # Delimiter patterns run on raw input (before tag stripping) so that
    # <system>…</system> tags are still present for matching.
    for pattern, vtype, label in _INJECTION_PATTERNS:
        text_to_check = user_input if vtype == ViolationType.DELIMITER_COLLISION else sanitized
        if pattern.search(text_to_check):
            violations.append(vtype)
            details[label] = f"Matched injection pattern '{label}'."
            risk_score = max(risk_score, 0.9)
            logger.warning("[Guardrail] Injection violation '%s' detected.", label)

    # ── Layer 2: Toxicity & Abuse ─────────────────────────────────────────
    for pattern, vtype, label in _TOXICITY_PATTERNS:
        if pattern.search(sanitized):
            violations.append(vtype)
            details[label] = f"Matched toxicity pattern '{label}'."
            risk_score = max(risk_score, 0.95)
            logger.warning("[Guardrail] Toxicity violation '%s' detected.", label)

    # ── Layer 3: Domain relevance ─────────────────────────────────────────
    is_clearly_off_domain = any(p.search(sanitized) for p in _OUT_OF_DOMAIN_PATTERNS)
    has_yoga_signal = bool(_YOGA_DOMAIN_SIGNALS.search(sanitized))

    if is_clearly_off_domain and not has_yoga_signal:
        violations.append(ViolationType.OUT_OF_DOMAIN)
        details["out_of_domain"] = (
            "Request appears unrelated to yoga, wellness, or health practice. "
            "YogaSutra only processes yoga and mindfulness-related queries."
        )
        risk_score = max(risk_score, 0.75)
        logger.info("[Guardrail] Out-of-domain request flagged.")

    # ── Determine final safety verdict ────────────────────────────────────
    is_safe = len(violations) == 0

    rejection_reason = ""
    if not is_safe:
        # Compose a user-friendly rejection message
        critical = {ViolationType.PROMPT_INJECTION, ViolationType.JAILBREAK,
                    ViolationType.ROLE_HIJACKING, ViolationType.DELIMITER_COLLISION,
                    ViolationType.TOXICITY, ViolationType.HATE_SPEECH,
                    ViolationType.SELF_HARM}
        has_critical = bool(set(violations) & critical)
        if has_critical:
            rejection_reason = (
                "Your request contains content that cannot be processed by YogaSutra. "
                "Please submit a genuine yoga or wellness question."
            )
        elif ViolationType.OUT_OF_DOMAIN in violations:
            rejection_reason = (
                "YogaSutra is a yoga and mindfulness assistant. "
                "Please ask a question related to yoga practice, poses, breathing, or wellness."
            )
        elif ViolationType.EXCESSIVE_LENGTH in violations:
            rejection_reason = (
                f"Your request is too long (maximum {_MAX_INPUT_CHARS} characters). "
                "Please shorten your query."
            )
        else:
            rejection_reason = "Request could not be processed for security reasons."

    result = GuardrailResult(
        is_safe=is_safe,
        risk_score=round(risk_score, 3),
        violations=list(dict.fromkeys(violations)),  # deduplicate, preserve order
        rejection_reason=rejection_reason,
        sanitized_input=sanitized if is_safe else "",
        details=details,
    )

    if not is_safe:
        logger.warning(
            "[Guardrail] BLOCKED request. risk_score=%.3f violations=%s",
            result.risk_score,
            [v.value for v in result.violations],
        )
    else:
        logger.debug(
            "[Guardrail] PASSED request. risk_score=%.3f length=%d",
            result.risk_score,
            len(sanitized),
        )

    return result
