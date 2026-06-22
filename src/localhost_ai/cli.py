from __future__ import annotations

import json
import logging
import sys

import typer

from .config import Settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="localhost-ai inference server")
models_app = typer.Typer(no_args_is_help=True, help="Pinned model files in .models/")
app.add_typer(models_app, name="models")


@app.command()
def serve(
    host: str = typer.Option(None, help="Bind address (default LHAI_HOST or 127.0.0.1)"),
    port: int = typer.Option(None, help="Port (default LHAI_PORT or 8000)"),
    model: str = typer.Option(None, help="Preset name from models.yaml"),
    device: str = typer.Option(None, help="auto | cuda | mps | cpu"),
    controller: str = typer.Option(None, help="aimd | fixed"),
    log_level: str = typer.Option("info"),
) -> None:
    """Load the model and serve the OpenAI-compatible, WebSocket and metrics APIs."""
    import uvicorn

    from .api.app import create_app
    from .service import build_from_settings

    overrides = {k: v for k, v in {"host": host, "port": port, "model": model, "device": device,
                                   "controller": controller}.items() if v is not None}
    s = Settings(**overrides)
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: "
                        "%(message)s")
    svc = build_from_settings(s)
    # One worker only: the engine and its KV cache live in this process.
    uvicorn.run(create_app(svc), host=s.host, port=s.port, log_level=log_level, workers=1)


@models_app.command("pull")
def models_pull(
    all_models: bool = typer.Option(False, "--all", help="Every preset, not just the default"),
    write_lock: bool = typer.Option(False, help="Record hashes in models.lock instead of "
                                                 "checking them"),
) -> None:
    """Download pinned revisions into .models/ and check them against models.lock."""
    from .models.registry import Registry, download, hashes, read_lock, verify
    from .models.registry import write_lock as wl

    s = Settings()
    reg = Registry(s.models_file)
    names = list(reg.specs) if all_models else [s.model]
    lock = read_lock(s.models_lock)
    bad = []
    for name in names:
        spec = reg.get(name)
        path = download(spec, s.models_dir)
        if write_lock:
            lock[spec.name] = {"repo": spec.repo, "revision": spec.revision,
                               "files": hashes(path)}
            typer.echo(f"locked {spec.name} ({spec.repo}@{spec.revision[:12]})")
        else:
            problems = verify(spec, path, lock)
            bad += problems
            typer.echo(f"{'ok' if not problems else 'MISMATCH'} {spec.name} -> {path}")
    if write_lock:
        wl(s.models_lock, lock)
    for p in bad:
        typer.echo(p, err=True)
    raise typer.Exit(1 if bad else 0)


@models_app.command("verify")
def models_verify() -> None:
    """Check the local snapshot of the default model against models.lock, offline."""
    from .models.registry import Registry, local_path, read_lock, verify

    s = Settings()
    spec = Registry(s.models_file).get(s.model)
    problems = verify(spec, local_path(spec, s.models_dir), read_lock(s.models_lock))
    for p in problems:
        typer.echo(p, err=True)
    typer.echo("ok" if not problems else "FAILED")
    raise typer.Exit(1 if problems else 0)


@app.command("egress-check")
def egress_check(
    expect: str = typer.Option("blocked", help="blocked: exit 0 only if every connect fails; "
                                              "open: exit 0 only if every connect succeeds"),
    timeout: float = 3.0,
) -> None:
    """Try to reach the internet from this process (the offline canary)."""
    from .egress import probe

    results = probe(timeout)
    typer.echo(json.dumps(results, indent=2))
    opened = [k for k, v in results.items() if v == "open"]
    ok = not opened if expect == "blocked" else len(opened) == len(results)
    raise typer.Exit(0 if ok else 1)


@app.command()
def info() -> None:
    """Print the device, dtype and memory probe this machine would use."""
    from . import device
    from .memory import probe_for

    s = Settings()
    dev = device.configure(s.device, s.dtype, s.threads)
    snap = probe_for(dev.kind, s.mem_limit_bytes).snapshot()
    typer.echo(json.dumps({"device": dev.kind, "dtype": dev.dtype_name, "threads": dev.threads,
                           "memory": {"used": snap.used, "limit": snap.limit,
                                      "headroom_frac": round(snap.headroom_frac, 3),
                                      "source": snap.source},
                           "python": sys.version.split()[0]}, indent=2))


if __name__ == "__main__":
    app()
