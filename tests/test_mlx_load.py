"""load_mlx end to end on a tiny 4-bit MLX checkpoint written to a temp dir (random weights, a
word-level tokenizer with a chat template), plus MlxProbe on the real allocator. Needs the mlx
extra, so it skips elsewhere."""

import json

import pytest

mx = pytest.importorskip("mlx.core")

import mlx.nn as nn  # noqa: E402
import torch  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from mlx_lm.models import llama  # noqa: E402

from localhost_ai.device import DeviceConfig  # noqa: E402
from localhost_ai.memory import MlxProbe  # noqa: E402
from localhost_ai.models import loader  # noqa: E402
from localhost_ai.models.registry import ModelSpec  # noqa: E402

WORDS = ["<eos>", "<user>", "<assistant>", "hello", "world", "paris", "is", "the"]
TEMPLATE = ("{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} {% endfor %}"
            "{% if add_generation_prompt %}<assistant>{% endif %}")


def write_checkpoint(path, bits=4):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    config = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
              "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
              "rms_norm_eps": 1e-5, "vocab_size": 128, "eos_token_id": 0,
              "quantization": {"group_size": 32, "bits": bits}}
    mx.random.seed(0)
    model = llama.Model(llama.ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=bits)
    mx.save_safetensors(str(path / "model.safetensors"), dict(tree_flatten(model.parameters())))
    (path / "config.json").write_text(json.dumps(config))
    vocab = {w: i for i, w in enumerate(WORDS)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<eos>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>",
                                   chat_template=TEMPLATE)
    fast.save_pretrained(path)


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    write_checkpoint(tmp_path)
    monkeypatch.setattr(loader, "local_path", lambda spec, models_dir: tmp_path)
    return tmp_path


def spec(**kw):
    return ModelSpec(name="tiny", repo="org/tiny", revision="0" * 40, backend="mlx", **kw)


def test_load_mlx_happy_path(checkpoint):
    dev = DeviceConfig(torch.device("cpu"), torch.float32, threads=1)  # unused by mlx
    m = loader.load(spec(quantization="mlx4"), dev, checkpoint.parent)
    assert m.quant == "mlx4" and m.dtype_name == "float32"
    assert m.runner.eos_ids == frozenset({0})
    assert m.runner.kv_bytes_per_token == 2 * 2 * 2 * 16 * 4  # 2 layers, K+V, 2 heads x 16, fp32
    ids = m.encode_chat([{"role": "user", "content": "hello world"}])
    assert ids == [1, 3, 4, 2]
    state, logits = m.runner.prefill([ids])
    assert logits.shape == (1, 128) and m.runner.kv_bytes(state) > 0


def test_load_mlx_rejects_a_wrong_quantization_label(checkpoint):
    dev = DeviceConfig(torch.device("cpu"), torch.float32, threads=1)
    with pytest.raises(RuntimeError, match="says mlx8, the checkpoint's config.json says mlx4"):
        loader.load(spec(quantization="mlx8"), dev, checkpoint.parent)


@pytest.mark.skipif(not mx.metal.is_available() or mx.default_device() != mx.gpu,
                    reason="MlxProbe reads the Metal allocator")
def test_mlx_probe_reads_the_allocator():
    probe = MlxProbe()
    before = probe.snapshot()
    a = mx.zeros((64, 1024, 1024), dtype=mx.float32)  # 256 MiB
    mx.eval(a)
    after = probe.snapshot()
    assert after.used - before.used >= 256 * 2**20
    assert 0 < after.limit and after.headroom <= after.limit - after.used + 1
    assert after.source
    del a
