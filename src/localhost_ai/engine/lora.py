"""Several LoRA adapters on one MLX base model, with a different adapter (or none) per batch row.

Every linear layer that any adapter touches is wrapped in a `MultiLoRALinear`. It keeps the
base layer as it is (4-bit weights stay quantized) and stacks the adapters' low-rank factors,
slot 0 being an all-zero "no adapter" slot:

    y = base(x) + scale[i] * (x @ A[i]) @ B[i]      for a row that uses adapter i

The runner tells the wrappers which adapter each row uses before every forward pass
(`AdapterSet.use`). Three cases, from cheapest:
- no row uses an adapter: the base layer only, so base-model requests cost nothing extra;
- every row uses the same adapter: the same unfused matmuls mlx-lm's LoRALinear runs;
- mixed rows: `mx.gather_mm` picks each row's factors inside one batched matmul, so rows with
  different adapters (and rows with none) share one forward pass.

Adapters are mlx-lm LoRA checkpoints: a directory with adapter_config.json and
adapters.safetensors (or a path to one checkpoint file inside such a directory). Adapters of
different ranks are zero-padded to the largest rank, which adds exact zeros. DoRA and full
fine-tunes aren't supported."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from .mlx_runner import MLXModelRunner


@dataclass
class AdapterSpec:
    name: str
    weights: Path
    config: dict

    @classmethod
    def from_path(cls, name: str, path: str | Path) -> AdapterSpec:
        p = Path(path).expanduser()
        weights = p / "adapters.safetensors" if p.is_dir() else p
        cfg_file = (p if p.is_dir() else p.parent) / "adapter_config.json"
        if not weights.exists():
            raise FileNotFoundError(f"adapter {name!r}: {weights} not found")
        if not cfg_file.exists():
            raise FileNotFoundError(f"adapter {name!r}: {cfg_file} not found")
        cfg = json.loads(cfg_file.read_text())
        kind = cfg.get("fine_tune_type", "lora")
        if kind != "lora":
            raise ValueError(f"adapter {name!r} is a {kind} fine-tune; only lora is supported")
        return cls(name, weights, cfg)

    @property
    def scale(self) -> float:
        return float(self.config.get("lora_parameters", {}).get("scale", 20.0))


@dataclass
class Rows:
    """Which adapter slot each batch row uses, shared by every wrapped layer."""

    mode: str = "none"  # "none", "single" or "mixed"
    single: int = 0
    index: Any = None  # mx.array [B] of slots, for "mixed"


class MultiLoRALinear(nn.Module):
    def __init__(self, base: nn.Module, a: list[Any], b: list[Any], scales: list[float],
                 rows: Rows) -> None:
        super().__init__()
        self.base = base
        self._a = a  # per slot [in, r], float32; slot 0 is zeros
        self._b = b  # per slot [r, out]
        self._a_stack = mx.stack(a)
        self._b_stack = mx.stack(b)
        self._scales = scales
        self._scale_arr = mx.array(scales, dtype=mx.float32)
        self._rows = rows

    def __call__(self, x: Any) -> Any:
        y = self.base(x)
        rows = self._rows
        if rows.mode == "none":
            return y
        if rows.mode == "single":
            i = rows.single
            z = (x @ self._a[i]) @ self._b[i]
            return y + (self._scales[i] * z).astype(x.dtype)
        # mixed: [B, T, in] x per-row [in, r] -> [B, T, r] x per-row [r, out] -> [B, T, out]
        idx = rows.index
        batch = mx.arange(x.shape[0], dtype=mx.uint32)
        z = mx.gather_mm(x.astype(mx.float32), self._a_stack, batch, idx)
        z = mx.gather_mm(z, self._b_stack, batch, idx)
        scale = self._scale_arr[idx].reshape(-1, 1, 1)
        return y + (scale * z).astype(x.dtype)


def _dims(layer: nn.Module) -> tuple[int, int]:
    out_dims, in_dims = layer.weight.shape
    if isinstance(layer, nn.QuantizedLinear):
        in_dims = in_dims * 32 // layer.bits
    return in_dims, out_dims


@dataclass
class AdapterSet:
    names: list[str]  # slot i + 1 is names[i]; slot 0 is the base model
    rows: Rows
    modules: list[str] = field(default_factory=list)

    def slot(self, name: str | None) -> int:
        if name is None:
            return 0
        try:
            return self.names.index(name) + 1
        except ValueError:
            raise KeyError(f"unknown adapter {name!r}; loaded: {', '.join(self.names)}") from None

    def use(self, slots: list[int]) -> None:
        """Set the per-row adapter slots for the next forward pass."""
        distinct = set(slots)
        if distinct <= {0}:
            self.rows.mode = "none"
        elif len(distinct) == 1:
            self.rows.mode, self.rows.single = "single", slots[0]
        else:
            self.rows.mode = "mixed"
            self.rows.index = mx.array(slots, dtype=mx.uint32)


def install(model: nn.Module, specs: list[AdapterSpec]) -> AdapterSet:
    """Wrap every layer any adapter touches; returns the set used to route rows."""
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("adapter names must be unique")
    factors: dict[str, dict[int, tuple[Any, Any]]] = {}
    for slot, spec in enumerate(specs, start=1):
        weights = mx.load(str(spec.weights))
        for key, arr in weights.items():
            if not key.endswith(".lora_a"):
                continue
            path = key.removesuffix(".lora_a")
            b = weights.get(f"{path}.lora_b")
            if b is None:
                raise ValueError(f"adapter {spec.name!r}: {path} has lora_a but no lora_b")
            factors.setdefault(path, {})[slot] = (arr.astype(mx.float32),
                                                  b.astype(mx.float32))
        if not any(slot in f for f in factors.values()):
            raise ValueError(f"adapter {spec.name!r}: no lora_a/lora_b weights in {spec.weights}")
    modules = dict(model.named_modules())
    rows = Rows()
    wrapped = []
    for path, per_slot in sorted(factors.items()):
        layer = modules.get(path)
        if not isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            raise ValueError(f"adapter weights for {path}, which is "
                             f"{type(layer).__name__ if layer is not None else 'missing'} "
                             "in this model")
        in_dims, out_dims = _dims(layer)
        rank = max(a.shape[1] for a, _ in per_slot.values())
        a_list = [mx.zeros((in_dims, rank), dtype=mx.float32)]
        b_list = [mx.zeros((rank, out_dims), dtype=mx.float32)]
        scales = [0.0]
        for slot, spec in enumerate(specs, start=1):
            if slot not in per_slot:
                a_list.append(a_list[0])
                b_list.append(b_list[0])
                scales.append(0.0)
                continue
            a, b = per_slot[slot]
            if a.shape[0] != in_dims or b.shape[1] != out_dims:
                raise ValueError(f"adapter {spec.name!r}: {path} is {a.shape[0]}x{b.shape[1]}, "
                                 f"the model's layer is {in_dims}x{out_dims}")
            r = a.shape[1]
            if r < rank:  # zero-pad to the widest rank; the padding contributes exact zeros
                a = mx.concatenate([a, mx.zeros((in_dims, rank - r), dtype=mx.float32)], 1)
                b = mx.concatenate([b, mx.zeros((rank - r, out_dims), dtype=mx.float32)], 0)
            a_list.append(a)
            b_list.append(b)
            scales.append(spec.scale)
        wrapped.append((path, MultiLoRALinear(layer, a_list, b_list, scales, rows)))
    model.update_modules(tree_unflatten(wrapped))
    mx.eval([(m._a_stack, m._b_stack) for _, m in wrapped])
    return AdapterSet([s.name for s in specs], rows, [p for p, _ in wrapped])


def _slots(state: Any) -> list[int]:
    # batches built inside MLXModelRunner (prefix-cache groups) have no slots yet: base rows
    return getattr(state, "slots", None) or [0] * len(state.lengths)


class LoRAMLXRunner(MLXModelRunner):
    """MLXModelRunner whose rows can each use a different adapter. The batch state carries one
    adapter slot per row (`state.slots`) through merge and select, and the slots are handed to
    the wrapped layers before every forward pass. Requests without an adapter run the base
    model unchanged, alone or in a batch with adapter rows.

    Stored prompt prefixes (engine/prefix.py) hold base-model KV, so a prefill with any adapter
    row skips the prefix cache; base-only prefills use it as usual."""

    def __init__(self, model: Any, eos_ids: frozenset[int], adapters: AdapterSet,
                 **kw: Any) -> None:
        self.adapters = adapters
        super().__init__(model, eos_ids, **kw)

    def prefill(self, seqs: list[list[int]], adapters: list[str | None] | None = None):
        slots = [self.adapters.slot(a) for a in adapters] if adapters else [0] * len(seqs)
        self.adapters.use(slots)
        if any(slots):
            state, logits = self._prefill(seqs, None)
        else:
            state, logits = super().prefill(seqs)
        state.slots = slots
        return state, logits

    def decode(self, state: Any, tokens: list[int]):
        self.adapters.use(_slots(state))
        return super().decode(state, tokens)

    def merge(self, a: Any, b: Any):
        out = super().merge(a, b)
        out.slots = _slots(a) + _slots(b)
        return out

    def select(self, state: Any, keep: list[int]):
        slots = _slots(state)
        out = super().select(state, keep)
        out.slots = [slots[i] for i in keep]
        return out
