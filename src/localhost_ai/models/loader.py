from __future__ import annotations

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
    runner: HFModelRunner
    tokenizer: Any
    dtype: torch.dtype
    path: Path
    load_s: float

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        out = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 tokenize=True)
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
                               f"this device is {dev.kind}")
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
    loaded = LoadedModel(spec, runner, tokenizer, dtype, path, time.perf_counter() - t)
    log.info("loaded %s@%s on %s/%s in %.1fs (kv %d B/token)", spec.repo, spec.revision[:8],
             dev.kind, str(dtype).removeprefix("torch."), loaded.load_s,
             runner.kv_bytes_per_token)
    return loaded


def _warmup(runner: HFModelRunner) -> None:
    state, logits = runner.prefill([[1, 2, 3, 4], [5, 6]])
    runner.decode(state, logits.argmax(-1).tolist())
