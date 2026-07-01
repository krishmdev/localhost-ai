from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..device import DeviceConfig, pick_dtype
from ..engine.runner import HFModelRunner
from .registry import ModelSpec, local_path

log = logging.getLogger("localhost_ai.models")


@dataclass
class LoadedModel:
    spec: ModelSpec
    runner: Any  # HFModelRunner or MLXModelRunner
    tokenizer: Any
    dtype: Any  # torch.dtype, or mlx.core.Dtype for the mlx backend
    path: Path
    load_s: float
    quant: str = "none"

    @property
    def dtype_name(self) -> str:
        return str(self.dtype).removeprefix("torch.").removeprefix("mlx.core.")

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        out = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=True,
                                                 **(self.spec.chat_template_kwargs or {}))
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            out = out["input_ids"]
        return list(out)


def _eos_ids(tokenizer: Any, model: Any) -> frozenset[int]:
    ids: set[int] = set()
    gen = getattr(model, "generation_config", None)
    raw = getattr(gen, "eos_token_id", None)
    if isinstance(raw, int):
        ids.add(raw)
    elif raw:
        ids.update(raw)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return frozenset(ids)


def load(spec: ModelSpec, dev: DeviceConfig, models_dir: Path, quantization: str = "none",
         dtype_override: str = "auto") -> LoadedModel:
    if spec.backend == "mlx":
        if quantization != "none":
            raise RuntimeError(f"quantization={quantization} applies to torch presets; "
                               f"{spec.name} is an MLX checkpoint that is already quantized")
        return load_mlx(spec, models_dir)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t = time.perf_counter()
    path = local_path(spec, models_dir)
    requested = dtype_override if dtype_override != "auto" else (spec.dtype or "auto")
    dtype = pick_dtype(dev.device, requested)
    if dev.kind == "cpu" and dtype != torch.float32:
        dtype = torch.float32  # half-precision matmuls on CPU are slow or unsupported
    kwargs: dict[str, Any] = {"dtype": dtype, "attn_implementation": "sdpa"}
    quant = quantization if quantization != "none" else (spec.quantization or "none")
    if quant != "none":
        if dev.kind != "cuda":
            raise RuntimeError(f"quantization={quant} needs bitsandbytes on CUDA; "
                               f"this device is {dev.kind}. On Apple silicon, use a preset "
                               f"with backend: mlx (a pre-quantized MLX checkpoint)")
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=quant == "bnb8", load_in_4bit=quant == "bnb4")
        kwargs["device_map"] = {"": 0}

    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    if quant == "none":
        model = model.to(dev.device)
    model.eval()
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    runner = HFModelRunner(model, pad_id=pad, eos_ids=_eos_ids(tokenizer, model),
                           device=dev.device, dtype=dtype)
    _warmup(runner)
    loaded = LoadedModel(spec, runner, tokenizer, dtype, path, time.perf_counter() - t, quant)
    log.info("loaded %s@%s on %s/%s in %.1fs (kv %d B/token)", spec.repo, spec.revision[:8],
             dev.kind, str(dtype).removeprefix("torch."), loaded.load_s,
             runner.kv_bytes_per_token)
    return loaded


def _warmup(runner: HFModelRunner) -> None:
    state, logits = runner.prefill([[1, 2, 3, 4], [5, 6]])
    runner.decode(state, logits.argmax(-1).tolist())


def mlx_bits(config: dict) -> int | None:
    """Weight bits of an MLX checkpoint from its config.json, None if it isn't quantized."""
    q = config.get("quantization") or config.get("quantization_config")
    return q.get("bits") if isinstance(q, dict) else None


def _eos_from_files(path: Path, tokenizer: Any) -> frozenset[int]:
    """transformers' _eos_ids reads the model's generation_config; an MLX model has none, so read
    the same fields from the checkpoint's JSON files."""
    ids: set[int] = set()
    for name in ("generation_config.json", "config.json"):
        f = path / name
        if not f.exists():
            continue
        cfg = json.loads(f.read_text())
        for raw in (cfg.get("eos_token_id"), (cfg.get("text_config") or {}).get("eos_token_id")):
            if isinstance(raw, int):
                ids.add(raw)
            elif raw:
                ids.update(raw)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return frozenset(ids)


def load_mlx(spec: ModelSpec, models_dir: Path) -> LoadedModel:
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from mlx_lm.utils import load_model
    except ImportError as exc:
        raise RuntimeError(f"{spec.name} needs the MLX backend (Apple silicon only): "
                           f"uv sync --extra mlx ({exc})") from exc
    from transformers import AutoTokenizer

    from ..engine.mlx_runner import MLXModelRunner

    t = time.perf_counter()
    path = local_path(spec, models_dir)
    config = json.loads((path / "config.json").read_text())
    bits = mlx_bits(config)
    quant = f"mlx{bits}" if bits else "none"
    if spec.quantization is not None and spec.quantization != quant:
        raise RuntimeError(f"{spec.name}: models.yaml says {spec.quantization}, "
                           f"the checkpoint's config.json says {quant}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    model, _ = load_model(path)
    model.eval()
    # activation dtype: the most common floating dtype among the unquantized parameters
    kinds = [a.dtype for _, a in tree_flatten(model.parameters())
             if mx.issubdtype(a.dtype, mx.floating)]
    dtype = max(set(kinds), key=kinds.count) if kinds else mx.float16
    runner = MLXModelRunner(model, eos_ids=_eos_from_files(path, tokenizer))
    loaded = LoadedModel(spec, runner, tokenizer, dtype, path, time.perf_counter() - t, quant)
    log.info("loaded %s@%s with mlx/%s %s in %.1fs (kv %d B/token, %d B fixed state per row, "
             "%.2f GiB active)", spec.repo, spec.revision[:8], quant, loaded.dtype_name,
             loaded.load_s, runner.kv_bytes_per_token, runner.row_state_bytes,
             mx.get_active_memory() / 2**30)
    return loaded
