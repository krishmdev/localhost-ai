"""Remove host details that don't belong in a public repo from committed results files:
process lists, container names, the lease command, and any local tool or temp paths. Keeps
chip, RAM, OS, power, library versions, device config, swap and idle numbers.

    uv run python scripts/scrub_manifests.py bench/results/*.json docs/results/*.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DROP_KEYS = {"top_cpu", "top_mem", "docker_ps"}
PATHY = re.compile(r"(/private/tmp|/Users/|\.tools|claude)", re.IGNORECASE)


def scrub(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in DROP_KEYS:
                continue
            if k == "compute_lease" and isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk in ("held", "holder", "holder_env")}
            if k in ("manifest", "out", "prompts") and isinstance(v, str) and PATHY.search(v):
                v = Path(v).name
            out[k] = scrub(v)
        return out
    if isinstance(obj, list):
        return [scrub(x) for x in obj]
    if isinstance(obj, str) and PATHY.search(obj):
        return re.sub(r"\S*(/private/tmp|/Users/)\S*", "<path>", obj)
    return obj


def main(paths: list[str]) -> None:
    for p in paths:
        path = Path(p)
        data = scrub(json.loads(path.read_text()))
        text = json.dumps(data, indent=1) + "\n"
        leftover = PATHY.findall(text)
        path.write_text(text)
        print(f"{p}: scrubbed{'; still contains ' + str(set(leftover)) if leftover else ''}")


if __name__ == "__main__":
    main(sys.argv[1:])
