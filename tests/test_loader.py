"""Loader paths that need no weights: quantization routing, MLX checkpoint metadata, chat
template kwargs."""

import json
from types import SimpleNamespace

import pytest
import torch

from localhost_ai.device import DeviceConfig
from localhost_ai.models import loader
from localhost_ai.models.loader import LoadedModel, _eos_from_files, mlx_bits
from localhost_ai.models.registry import ModelSpec

CPU = DeviceConfig(torch.device("cpu"), torch.float32, threads=1)


def spec(**kw):
    return ModelSpec(name="m", repo="org/m", revision="0" * 40, license="apache-2.0", **kw)


def test_mlx_bits_from_config():
    assert mlx_bits({"quantization": {"group_size": 64, "bits": 4}}) == 4
    assert mlx_bits({"quantization_config": {"group_size": 64, "bits": 8}}) == 8
    assert mlx_bits({"torch_dtype": "bfloat16"}) is None


def test_eos_from_checkpoint_files(tmp_path):
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"eos_token_id": 3}}))
    tok = SimpleNamespace(eos_token_id=4)
    assert _eos_from_files(tmp_path, tok) == {1, 2, 3, 4}
    assert _eos_from_files(tmp_path / "missing", SimpleNamespace(eos_token_id=None)) == set()


class RecordingTokenizer:
    def __init__(self):
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.kwargs = kwargs
        return {"input_ids": [7, 8, 9]}


def test_encode_chat_passes_preset_template_kwargs(tmp_path):
    tok = RecordingTokenizer()
    m = LoadedModel(spec(backend="mlx", chat_template_kwargs={"enable_thinking": False}),
                    runner=None, tokenizer=tok, dtype=torch.float16, path=tmp_path, load_s=0.0)
    assert m.encode_chat([{"role": "user", "content": "hi"}]) == [7, 8, 9]
    assert tok.kwargs["enable_thinking"] is False and tok.kwargs["add_generation_prompt"]
    plain = LoadedModel(spec(), None, RecordingTokenizer(), torch.float32, tmp_path, 0.0)
    plain.encode_chat([{"role": "user", "content": "hi"}])
    assert "enable_thinking" not in plain.tokenizer.kwargs


def test_dtype_name_strips_the_framework_prefix(tmp_path):
    assert LoadedModel(spec(), None, None, torch.bfloat16, tmp_path, 0.0).dtype_name == "bfloat16"
    mlx_dtype = type("Dtype", (), {"__str__": lambda self: "mlx.core.float16"})()
    assert LoadedModel(spec(), None, None, mlx_dtype, tmp_path, 0.0).dtype_name == "float16"


def test_bitsandbytes_off_cuda_points_to_mlx_presets(tmp_path, monkeypatch):
    monkeypatch.setattr(loader, "local_path", lambda s, d: tmp_path)
    with pytest.raises(RuntimeError, match="backend: mlx"):
        loader.load(spec(quantization="bnb4"), CPU, tmp_path)
    with pytest.raises(RuntimeError, match="needs bitsandbytes on CUDA"):
        loader.load(spec(), CPU, tmp_path, quantization="bnb8")


def test_mlx_preset_rejects_a_bitsandbytes_override(tmp_path):
    with pytest.raises(RuntimeError, match="already quantized"):
        loader.load(spec(backend="mlx", quantization="mlx4"), CPU, tmp_path, quantization="bnb4")


def test_mlx_label_must_match_the_checkpoint(tmp_path, monkeypatch):
    pytest.importorskip("mlx_lm")
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 8}}))
    monkeypatch.setattr(loader, "local_path", lambda s, d: tmp_path)
    with pytest.raises(RuntimeError, match="models.yaml says mlx4.*config.json says mlx8"):
        loader.load(spec(backend="mlx", quantization="mlx4"), CPU, tmp_path)
