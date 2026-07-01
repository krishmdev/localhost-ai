"""Model presets from models.yaml, pinned downloads into a project-local cache, and the
sha256 lock that makes a fresh machine reproduce exactly the files this repo was tested with."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

# Only the files needed to run the model; the Hub repos also carry ONNX exports and training logs.
# Larger checkpoints are sharded (index + model-0000k-of-0000n), and newer tokenizers keep the
# chat template in its own file.
ALLOW = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
         "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt",
         "model.safetensors.index.json", "model-*-of-*.safetensors", "added_tokens.json",
         "chat_template.jinja"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    revision: str
    license: str = ""
    dtype: str | None = None
    quantization: str | None = None
    max_new_tokens: int = 256


class Registry:
    def __init__(self, path: Path) -> None:
        raw = yaml.safe_load(path.read_text())
        self.default: str = raw["default"]
        self.specs = {name: ModelSpec(name=name, **cfg) for name, cfg in raw["models"].items()}

    def get(self, name: str | None = None) -> ModelSpec:
        key = name or self.default
        if key in self.specs:
            return self.specs[key]
        for spec in self.specs.values():  # also accept the Hub repo id
            if spec.repo == key:
                return spec
        raise KeyError(f"unknown model {key!r}; known: {', '.join(self.specs)}")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local_path(spec: ModelSpec, models_dir: Path) -> Path:
    """Resolve the pinned snapshot from the local cache only. Never touches the network."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec.repo, revision=spec.revision, cache_dir=models_dir / "hub",
                                  allow_patterns=ALLOW, local_files_only=True))


def download(spec: ModelSpec, models_dir: Path) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(spec.repo, revision=spec.revision, cache_dir=models_dir / "hub",
                                  allow_patterns=ALLOW))


def hashes(snapshot: Path) -> dict[str, str]:
    return {p.name: _sha256(p) for p in sorted(snapshot.iterdir()) if p.is_file()}


def read_lock(lock: Path) -> dict:
    return json.loads(lock.read_text()) if lock.exists() else {}


def write_lock(lock: Path, entries: dict) -> None:
    lock.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")


def verify(spec: ModelSpec, snapshot: Path, lock: dict) -> list[str]:
    """Return a list of problems (empty when the snapshot matches the lock)."""
    entry = lock.get(spec.name)
    if entry is None:
        return [f"{spec.name}: not in models.lock"]
    problems = []
    if entry["revision"] != spec.revision:
        problems.append(f"{spec.name}: lock pins {entry['revision']}, models.yaml {spec.revision}")
    got = hashes(snapshot)
    for fname, digest in entry["files"].items():
        if got.get(fname) != digest:
            problems.append(f"{spec.name}/{fname}: sha256 mismatch or missing")
    return problems
