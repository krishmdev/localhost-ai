from fnmatch import fnmatch
from pathlib import Path

import pytest

from localhost_ai.models.registry import ALLOW, ModelSpec, Registry

ROOT = Path(__file__).resolve().parents[1]


def test_allow_list_fetches_sharded_checkpoints_and_nothing_extra():
    # huggingface_hub matches allow_patterns with fnmatch
    def wanted(name):
        return any(fnmatch(name, p) for p in ALLOW)

    for name in ["model.safetensors", "model-00001-of-00002.safetensors",
                 "model.safetensors.index.json", "chat_template.jinja", "added_tokens.json"]:
        assert wanted(name), name
    for name in ["onnx/model.onnx", "README.md", "runs/events.out.tfevents", "model.gguf"]:
        assert not wanted(name), name


def spec(**kw):
    return ModelSpec(name="m", repo="org/m", revision="0" * 40, license="apache-2.0", **kw)


def test_backend_and_quantization_must_match():
    assert spec().backend == "torch"
    assert spec(quantization="bnb4").quantization == "bnb4"
    assert spec(backend="mlx", quantization="mlx4").quantization == "mlx4"
    assert spec(backend="mlx").quantization is None  # an unquantized MLX checkpoint
    with pytest.raises(ValueError, match="not valid for backend mlx"):
        spec(backend="mlx", quantization="bnb4")  # bitsandbytes is CUDA/torch only
    with pytest.raises(ValueError, match="not valid for backend torch"):
        spec(quantization="mlx4")
    with pytest.raises(ValueError, match="backend must be one of"):
        spec(backend="gguf")


def test_registry_reads_backend_and_chat_template_kwargs(tmp_path):
    f = tmp_path / "models.yaml"
    f.write_text("""
default: small
models:
  small:
    repo: org/small
    revision: abc
    license: mit
  hybrid-mlx4:
    repo: org/hybrid-4bit
    revision: def
    license: apache-2.0
    backend: mlx
    quantization: mlx4
    chat_template_kwargs: {enable_thinking: false}
""")
    reg = Registry(f)
    assert reg.get().backend == "torch" and reg.get().chat_template_kwargs is None
    h = reg.get("org/hybrid-4bit")
    assert (h.backend, h.quantization) == ("mlx", "mlx4")
    assert h.chat_template_kwargs == {"enable_thinking": False}


def test_shipped_models_yaml_is_valid():
    reg = Registry(ROOT / "models.yaml")
    assert reg.default in reg.specs
    for s in reg.specs.values():
        assert len(s.revision) == 40, f"{s.name} must pin a full commit sha"
