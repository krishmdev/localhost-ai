import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

if os.environ.get("LHAI_MLX_DEVICE") == "cpu":
    # CI runners have no usable Metal device; the tiny-model MLX tests run on MLX's CPU backend
    try:
        import mlx.core as mx
    except ImportError:
        pass
    else:
        mx.set_default_device(mx.cpu)
