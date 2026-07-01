from fnmatch import fnmatch

from localhost_ai.models.registry import ALLOW


def test_allow_list_fetches_sharded_checkpoints_and_nothing_extra():
    # huggingface_hub matches allow_patterns with fnmatch
    def wanted(name):
        return any(fnmatch(name, p) for p in ALLOW)

    for name in ["model.safetensors", "model-00001-of-00002.safetensors",
                 "model.safetensors.index.json", "chat_template.jinja", "added_tokens.json"]:
        assert wanted(name), name
    for name in ["onnx/model.onnx", "README.md", "runs/events.out.tfevents", "model.gguf"]:
        assert not wanted(name), name
