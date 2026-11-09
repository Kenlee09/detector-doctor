"""Provider-agnostic LLM boundary.

Two implementations ship here:

* `MockLLM` — a deterministic stand-in. It is not a fake model, it is a *policy*: given
  the same structured context it always makes the same choice, which is what lets the
  whole pipeline run offline in CI and lets a test assert on the plan.
* `OpenAICompatLLM` — any OpenAI-compatible endpoint. Imported lazily so the core has
  no runtime dependencies.

Both speak one contract: `complete(system, user, schema)` returns a response whose
`data` is JSON when a schema was requested. Swapping them changes the reasoning quality,
never the surrounding invariants.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ..errors import ConfigError, SchemaError

JSON_BLOCK = re.compile(r"```json\s*(.+?)\s*```", re.DOTALL)


@dataclass
class LLMResponse:
    text: str
    data: dict[str, Any] | None = None
    tokens: int = 0
    model: str = "unknown"
    cache_hit: bool = False


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def complete(
        self, system: str, user: str, schema: Mapping[str, Any] | None = None
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """Cheap upper-ish bound: ~3.5 chars per token for mixed CJK/latin text."""
    return max(1, len(text) // 3)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply."""
    match = JSON_BLOCK.search(text)
    candidate = match.group(1) if match else text
    candidate = candidate.strip()
    if candidate.startswith("{"):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SchemaError(f"model output was not valid JSON: {exc}") from exc
    raise SchemaError("model output contained no JSON object")


def check_schema(data: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    """Minimal structural check: required keys and top-level types only.

    Deliberately not a full JSON-Schema implementation — the point is to reject the
    failure modes that actually occur (missing key, wrong container type), fast.
    """
    for key in schema.get("required", []):
        if key not in data:
            raise SchemaError(f"missing required key: {key}")
    types = schema.get("properties", {})
    py = {"array": list, "object": dict, "string": str, "number": (int, float), "boolean": bool}
    for key, spec in types.items():
        if key in data:
            expected = py.get(spec.get("type", "object") if isinstance(spec, dict) else "object")
            if expected and not isinstance(data[key], expected):
                raise SchemaError(f"key {key!r} has type {type(data[key]).__name__}, expected {spec}")


class MockLLM:
    """Deterministic reasoning stand-in.

    Reads the fenced JSON context the agents embed in the user message and applies a
    fixed, documented policy. Never used to *compute* a metric — only to pick options
    and to shape prose.
    """

    name = "mock"

    def complete(
        self, system: str, user: str, schema: Mapping[str, Any] | None = None
    ) -> LLMResponse:
        context = self._context(user)
        tokens = estimate_tokens(system) + estimate_tokens(user)

        if schema is not None and schema.get("properties", {}).get("selected") is not None:
            data = self._select(context)
            return LLMResponse(text=json.dumps(data, ensure_ascii=False), data=data, tokens=tokens, model=self.name)

        if schema is not None and schema.get("properties", {}).get("narrative") is not None:
            data = {"narrative": self._narrate(context)}
            return LLMResponse(text=json.dumps(data, ensure_ascii=False), data=data, tokens=tokens, model=self.name)

        return LLMResponse(text=self._narrate(context), tokens=tokens, model=self.name)

    # -- policy --------------------------------------------------------------------
    def _select(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Pick the best-value option per finding, up to `per_finding`, within budget."""
        per_finding = int(context.get("per_finding", 2))
        gpu_limit = float(context.get("gpu_minutes_available", 0.0))
        findings = context.get("findings", [])
        candidates = context.get("candidates", {})

        selected: list[dict[str, Any]] = []
        notes: list[str] = []
        spent = 0.0
        for finding in findings:
            kind = finding.get("kind")
            options = list(candidates.get(kind, []))
            options.sort(key=lambda o: (0 if o.get("cost") == "free" else 1, -o.get("expected_gain", 0.0) / max(o.get("gpu_minutes", 0.1), 0.1)))
            taken = 0
            for opt in options:
                if taken >= per_finding:
                    break
                cost = float(opt.get("gpu_minutes", 0.0))
                if spent + cost > gpu_limit:
                    notes.append(f"skipped {opt['id']}: would exceed remaining GPU budget")
                    continue
                spent += cost
                taken += 1
                selected.append(
                    {
                        "finding_id": finding.get("id"),
                        "finding_kind": kind,
                        "option_id": opt.get("id"),
                        "action": opt.get("action"),
                        "overrides": self._overrides(opt),
                        "gpu_minutes": cost,
                        "why": (
                            f"{opt.get('layer')} layer, +{opt.get('expected_gain')} AP50 expected "
                            f"for {cost} GPU-min: {opt.get('rationale')}"
                        ),
                        "source": opt.get("source", ""),
                    }
                )
        if not selected:
            notes.append("no option fit the remaining budget")
        return {"selected": selected, "notes": notes}

    @staticmethod
    def _overrides(opt: Mapping[str, Any]) -> dict[str, str]:
        """`override key=value` encoded in the playbook source becomes the override map."""
        source = str(opt.get("source", ""))
        token = source.split("override", 1)[-1].strip()
        if "=" in token:
            key, value = token.split("=", 1)
            return {key.strip(): value.strip()}
        return {}

    def _narrate(self, context: Mapping[str, Any]) -> str:
        findings = context.get("findings", [])
        plan = context.get("plan", [])
        if not findings and not plan:
            return "No robustness anomaly exceeded reporting thresholds."
        lines = []
        for f in findings:
            lines.append(f"- [{f.get('severity')}] {f.get('summary')}")
        if plan:
            lines.append("")
            lines.append("Recommended next steps:")
            for p in plan:
                lines.append(f"- {p.get('action')} (+{p.get('expected_gain', '?')} AP50 expected)")
        return "\n".join(lines)

    @staticmethod
    def _context(user: str) -> dict[str, Any]:
        match = JSON_BLOCK.search(user)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}


class OpenAICompatLLM:
    """Any OpenAI-compatible chat endpoint (OpenAI, DeepSeek, vLLM, Ollama, ...)."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        self.model = model or os.environ.get("DD_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("DD_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("DD_BASE_URL")
        self.temperature = temperature
        self.name = f"openai-compat:{self.model}"
        if not self.api_key:
            raise ConfigError("no API key: set DD_API_KEY or OPENAI_API_KEY, or use --llm mock")

    def complete(
        self, system: str, user: str, schema: Mapping[str, Any] | None = None
    ) -> LLMResponse:
        try:
            from openai import OpenAI  # imported lazily: optional dependency
        except ImportError as exc:  # pragma: no cover
            raise ConfigError("pip install 'detector-doctor[llm]' to use a real model") from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or estimate_tokens(text))
        data = extract_json(text) if schema is not None else None
        return LLMResponse(text=text, data=data, tokens=tokens, model=self.model)


def build_llm(mode: str = "mock", **kwargs: Any) -> LLMClient:
    if mode == "mock":
        return MockLLM()
    if mode in {"openai", "compat", "real"}:
        return OpenAICompatLLM(**kwargs)
    raise ConfigError(f"unknown llm mode: {mode!r} (expected 'mock' or 'openai')")
