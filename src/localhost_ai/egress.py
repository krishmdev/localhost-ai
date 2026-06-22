"""Egress canary: try to open TCP connections to a few public hosts from this process. Used by
`lhai egress-check` and by GET /v1/admin/egress, so the check runs inside the server process
itself, which is the process that must not reach the network."""

from __future__ import annotations

import socket

TARGETS = [("1.1.1.1", 443), ("api.openai.com", 443), ("huggingface.co", 443)]


def probe(timeout: float = 3.0) -> dict[str, str]:
    results = {}
    for host, port in TARGETS:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            results[f"{host}:{port}"] = "open"
        except OSError as exc:
            results[f"{host}:{port}"] = f"blocked ({type(exc).__name__}: {exc.strerror or exc})"
    return results
