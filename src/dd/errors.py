"""Error taxonomy.

Every error carries a stable `code` so traces and reports can be diffed across runs.
"""

from __future__ import annotations


class DDError(Exception):
    """Base class for all Detector Doctor errors."""

    code = "dd.error"


class ConfigError(DDError):
    code = "dd.config"


class BudgetExceeded(DDError):
    """Raised before an expensive side effect, never after.

    The check happens in `Budget.reserve_*`, which is called by the tool *before*
    it enqueues any GPU work. A rejected reservation must leave the run unchanged.
    """

    code = "dd.budget_exceeded"


class EvidenceMissing(DDError):
    """A claim referenced evidence that does not exist on disk."""

    code = "dd.evidence_missing"


class UnverifiedClaim(DDError):
    """A claim reached the report writer without passing verification."""

    code = "dd.unverified_claim"


class ToolError(DDError):
    code = "dd.tool"


class SchemaError(DDError):
    """Structured LLM output did not match the requested schema."""

    code = "dd.schema"
