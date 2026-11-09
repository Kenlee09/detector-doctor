"""Detector Doctor — a controlled agent for object-detector robustness diagnosis.

The package is intentionally split into a deterministic core (`tools`, `orchestrator`,
`verify`) and a narrow reasoning layer (`agents`). The LLM never computes a number;
it only ranks remediation options and writes prose over structured tool output.
"""

__version__ = "0.1.0"
