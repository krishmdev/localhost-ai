"""Multi-LoRA on a tiny random mlx-lm Llama: random adapters written in mlx-lm's format, checked
against mlx-lm's own LoRALinear and against each row run alone. Needs the mlx extra (Apple
silicon); skips elsewhere. Rows are teacher-forced, as in test_mlx_runner."""

import json

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.tuner.utils import load_adapters  # noqa: E402
from test_mlx_runner import FORCED, PROMPTS, VOCAB, assert_close, tiny_llama  # noqa: E402

from localhost_ai.engine.controller import FixedController  # noqa: E402
from localhost_ai.engine.lora import (  # noqa: E402
    AdapterSpec,
    LoRAMLXRunner,
    install,
)
from localhost_ai.engine.mlx_runner import MLXModelRunner  # noqa: E402
from localhost_ai.engine.request import Request, SamplingParams  # noqa: E402
from localhost_ai.engine.scheduler import Scheduler  # noqa: E402

PROJ = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def write_adapter(path, seed, rank, scale, layers=(0, 1), projections=PROJ):
    """An mlx-lm style LoRA adapter with random, clearly non-zero factors."""
    model = tiny_llama()
    mods = dict(model.named_modules())
    mx.random.seed(seed)
    weights = {}
    for i in layers:
        for p in projections:
            out_dims, in_dims = mods[f"model.layers.{i}.{p}"].weight.shape
            weights[f"model.layers.{i}.{p}.lora_a"] = mx.random.normal((in_dims, rank)) * 0.1
            weights[f"model.layers.{i}.{p}.lora_b"] = mx.random.normal((rank, out_dims)) * 0.1
    path.mkdir(parents=True)
    mx.save_safetensors(str(path / "adapters.safetensors"), weights)
    keys = sorted(projections) if list(projections) != PROJ else None
    params = {"rank": rank, "scale": scale, "dropout": 0.0, **({"keys": keys} if keys else {})}
    (path / "adapter_config.json").write_text(json.dumps(
        {"fine_tune_type": "lora", "num_layers": len(layers), "lora_parameters": params}))
    return path


@pytest.fixture(scope="module")
def adapters(tmp_path_factory):
    root = tmp_path_factory.mktemp("adapters")
    return {"tutor": write_adapter(root / "tutor", 1, 8, 20.0),
            # lower rank, last layer and attention only, different scale
            "terse": write_adapter(root / "terse", 2, 4, 5.0, layers=(1,),
                                   projections=PROJ[:4])}


@pytest.fixture(scope="module")
def runner(adapters):
    model = tiny_llama()
    aset = install(model, [AdapterSpec.from_path(n, p) for n, p in adapters.items()])
    return LoRAMLXRunner(model, eos_ids=frozenset({0}), adapters=aset)


@pytest.fixture(scope="module")
def plain():
    return MLXModelRunner(tiny_llama(), eos_ids=frozenset({0}))


def alone(r, prompt, forced, adapter=None):
    state, logits = (r.prefill([prompt], adapters=[adapter]) if adapter
                     else r.prefill([prompt]))
    out = [logits[0]]
    for t in forced:
        out.append(r.decode(state, [t])[0])
    return out


def full_logits(model, seq):
    out = model(mx.array([seq], dtype=mx.int32))[0, -1].astype(mx.float32)
    return torch.from_numpy(np.array(out))


@pytest.mark.parametrize("name", ["tutor", "terse"])
def test_adapter_row_matches_mlx_lm_lora(runner, adapters, name):
    ref = load_adapters(tiny_llama(), str(adapters[name]))
    got = alone(runner, PROMPTS[0], FORCED[0], name)
    seq = list(PROMPTS[0])
    for step, t in enumerate([None, *FORCED[0]]):
        if t is not None:
            seq.append(t)
        assert_close(got[step], full_logits(ref, seq))


def test_adapters_change_the_output(runner, plain):
    base = alone(plain, PROMPTS[1], FORCED[1])[-1]
    for name in ("tutor", "terse"):
        assert (alone(runner, PROMPTS[1], FORCED[1], name)[-1] - base).abs().max() > 1e-2


def test_base_rows_are_untouched(runner, plain):
    for p, f in zip(PROMPTS, FORCED, strict=True):
        for a, b in zip(alone(runner, p, f), alone(plain, p, f), strict=True):
            assert torch.equal(a, b)  # no adapter row runs the base layers only


ROUTE = ["tutor", None, "terse", "tutor"]


def test_mixed_batch_matches_each_row_alone(runner):
    ref = [alone(runner, p, f, a) for p, f, a in zip(PROMPTS, FORCED, ROUTE, strict=True)]
    state, logits = runner.prefill(PROMPTS, adapters=ROUTE)
    steps = [logits]
    for i in range(len(FORCED[0])):
        steps.append(runner.decode(state, [f[i] for f in FORCED]))
    for step, got in enumerate(steps):
        for row in range(len(PROMPTS)):
            assert_close(got[row], ref[row][step])


def test_merge_and_select_keep_rows_on_their_adapter(runner):
    ref = [alone(runner, p, f, a) for p, f, a in zip(PROMPTS, FORCED, ROUTE, strict=True)]
    a, _ = runner.prefill(PROMPTS[:2], adapters=ROUTE[:2])
    runner.decode(a, [FORCED[0][0], FORCED[1][0]])
    b, _ = runner.prefill([PROMPTS[2] + [FORCED[2][0]], PROMPTS[3] + [FORCED[3][0]]],
                          adapters=ROUTE[2:])
    state = runner.merge(a, b)
    assert state.slots == [runner.adapters.slot(x) for x in ROUTE]
    state = runner.select(state, [3, 0, 2])
    got = runner.decode(state, [FORCED[3][1], FORCED[0][1], FORCED[2][1]])
    for row, orig in enumerate([3, 0, 2]):
        assert_close(got[row], ref[orig][2])


def test_scheduler_routes_requests_to_their_adapter(runner):
    params = SamplingParams(temperature=0.0, max_tokens=8)

    def run(reqs, limit):
        s = Scheduler(runner, _Tok(), FixedController(limit))
        for r in reqs:
            s.add(r)
        while s.has_work():
            s.step()
        return [r.generated for r in reqs]

    batched = run([Request(p, params, adapter=a) for p, a in zip(PROMPTS, ROUTE, strict=True)],
                  limit=4)
    single = [run([Request(p, params, adapter=a)], 1)[0]
              for p, a in zip(PROMPTS, ROUTE, strict=True)]
    assert batched == single


class _Tok:
    eos_token_id = 0

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + i % 26) for i in ids)


def test_unknown_adapter_and_bad_files(runner, adapters, tmp_path):
    with pytest.raises(KeyError, match="unknown adapter"):
        runner.prefill([[1, 2]], adapters=["nope"])
    with pytest.raises(FileNotFoundError):
        AdapterSpec.from_path("x", tmp_path / "missing")
    dora = write_adapter(tmp_path / "dora", 3, 4, 1.0)
    cfg = json.loads((dora / "adapter_config.json").read_text())
    (dora / "adapter_config.json").write_text(json.dumps({**cfg, "fine_tune_type": "dora"}))
    with pytest.raises(ValueError, match="only lora"):
        AdapterSpec.from_path("x", dora)
    wide = tmp_path / "wide"
    wide.mkdir()
    mx.save_safetensors(str(wide / "adapters.safetensors"), {
        "model.layers.0.self_attn.q_proj.lora_a": mx.zeros((99, 4)),
        "model.layers.0.self_attn.q_proj.lora_b": mx.zeros((4, 64))})
    (wide / "adapter_config.json").write_text(json.dumps({"lora_parameters": {"scale": 1.0}}))
    with pytest.raises(ValueError, match="the model's layer is 64x64"):
        install(tiny_llama(), [AdapterSpec.from_path("wide", wide)])


def test_checkpoint_file_path_reads_the_config_next_to_it(adapters):
    spec = AdapterSpec.from_path("ckpt", adapters["tutor"] / "adapters.safetensors")
    assert spec.scale == 20.0 and spec.weights.name == "adapters.safetensors"


def test_vocab_is_unchanged(runner):
    _, logits = runner.prefill([[1, 2, 3]], adapters=["tutor"])
    assert logits.shape == (1, VOCAB)
    assert len(tree_flatten(runner.model.parameters())) > 0
