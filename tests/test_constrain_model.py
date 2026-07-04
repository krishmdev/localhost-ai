"""response_format on real models and real tokenizers.

- Tokenizer tests (default suite): the llguidance grammar built from each downloaded preset's
  tokenizer accepts real JSON tokenizations and masks out text that can't start a JSON object.
  Presets that aren't downloaded are skipped; no weights are loaded.
- Generation tests: SmolLM2-135M on CPU (`-m model`) and Qwen2.5-0.5B 4-bit on MLX
  (`-m mlx_model`) answer a batch of mixed prompts under a JSON schema; every answer that ends
  with `stop` parses and has the required fields, and unconstrained rows batched with them
  produce the same tokens as they do alone."""

import json

import pytest
import torch

from localhost_ai.config import get_settings
from localhost_ai.device import DeviceConfig
from localhost_ai.engine.constrain import Grammars
from localhost_ai.engine.controller import FixedController
from localhost_ai.engine.request import Request, SamplingParams
from localhost_ai.engine.scheduler import Scheduler

pytest.importorskip("llguidance")

SCHEMA = {"type": "json_schema", "json_schema": {"name": "city", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["city", "country", "population"],
    "properties": {"city": {"type": "string"}, "country": {"type": "string"},
                   # bounded: a small model otherwise repeats digits until max_tokens
                   "population": {"type": "integer", "minimum": 0, "maximum": 10**9}}}}}
PROMPTS = ["Describe Paris as JSON with city, country and population.",
           "Give me facts about Tokyo.",
           "Hello",
           "Write a haiku about rain."]


def snapshot(name):
    from localhost_ai.models.registry import Registry, local_path

    s = get_settings()
    try:
        return local_path(Registry(s.models_file).get(name), s.models_dir)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{name} not downloaded ({exc})")


@pytest.mark.parametrize("preset", ["smollm2-135m", "qwen2.5-0.5b-mlx4", "qwen2.5-3b-mlx4",
                                    "llama-3.2-3b-mlx4", "gemma-4-e4b-mlx4", "qwen3.5-9b-mlx4"])
def test_real_tokenizer_grammar(preset):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(snapshot(preset))
    grammars = Grammars.from_hf(tok, frozenset({tok.eos_token_id}))
    good = '{"city": "Paris", "country": "France", "population": 2102650}'
    c = grammars.constraint(SCHEMA)
    for t in tok.encode(good, add_special_tokens=False):
        assert c.allowed(len(tok))[t], tok.decode([t])
        c.advance(t)
    assert c.allowed(len(tok))[tok.eos_token_id] and not c.failed
    fresh = grammars.constraint({"type": "json_object"})
    allowed = fresh.allowed(len(tok) + 64)
    assert not allowed[len(tok):].any()  # ids past the tokenizer's vocab never
    for word in ("Hello", " Sure", "```"):
        assert not allowed[tok.encode(word, add_special_tokens=False)[0]], word


def generate(loaded, eos):
    grammars = Grammars.from_hf(loaded.tokenizer, eos)
    sched = Scheduler(loaded.runner, loaded.tokenizer, FixedController(8))
    params = SamplingParams(temperature=0.0, max_tokens=64)
    shaped, plain = [], []
    for p in PROMPTS:
        ids = loaded.encode_chat([{"role": "user", "content": p}])
        r = Request(ids, params, constraint=grammars.constraint(SCHEMA))
        q = Request(ids, params)
        shaped.append(r)
        plain.append(q)
        sched.add(r)
        sched.add(q)
    while sched.has_work():
        sched.step()
    alone = []
    for q in plain:
        s = Scheduler(loaded.runner, loaded.tokenizer, FixedController(1))
        a = Request(q.prompt_ids, params)
        s.add(a)
        while s.has_work():
            s.step()
        alone.append(a.generated)
    return shaped, plain, alone


def check(loaded, shaped, plain, alone):
    horizon = 24 if getattr(loaded, "model_id", "").endswith("-mlx4") else 64
    assert [q.generated[:horizon] for q in plain] == [a[:horizon] for a in alone]
    stopped = 0
    for r in shaped:
        body = loaded.tokenizer.decode(r.generated, skip_special_tokens=True)
        if r.finish_reason == "stop":
            stopped += 1
            obj = json.loads(body)
            assert set(obj) == {"city", "country", "population"}, body
            assert isinstance(obj["population"], int)
    assert stopped >= len(shaped) - 1  # at most one runs out of its 64 tokens


@pytest.mark.model
def test_smollm2_json_schema_batched():
    from localhost_ai.models.loader import load
    from localhost_ai.models.registry import Registry

    s = get_settings()
    snapshot("smollm2-135m")
    dev = DeviceConfig(torch.device("cpu"), torch.float32, threads=torch.get_num_threads())
    loaded = load(Registry(s.models_file).get("smollm2-135m"), dev, s.models_dir)
    check(loaded, *generate(loaded, loaded.runner.eos_ids))


@pytest.mark.mlx_model
def test_qwen_mlx_json_schema_batched():
    pytest.importorskip("mlx_lm")
    from localhost_ai.models.loader import load
    from localhost_ai.models.registry import Registry

    s = get_settings()
    snapshot("qwen2.5-0.5b-mlx4")
    dev = DeviceConfig(torch.device("cpu"), torch.float32, threads=torch.get_num_threads())
    loaded = load(Registry(s.models_file).get("qwen2.5-0.5b-mlx4"), dev, s.models_dir)
    check(loaded, *generate(loaded, loaded.runner.eos_ids))
