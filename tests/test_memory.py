from localhost_ai.device import cgroup_cpu_limit
from localhost_ai.memory import BudgetProbe, CpuProbe, FakeProbe, MemSnapshot


def test_cgroup_probe_subtracts_inactive_file(tmp_path):
    (tmp_path / "memory.max").write_text("1000000\n")
    (tmp_path / "memory.current").write_text("600000\n")
    (tmp_path / "memory.stat").write_text("anon 400000\ninactive_file 100000\nactive_file 5\n")
    s = CpuProbe(cgroup_root=tmp_path).snapshot()
    assert (s.used, s.limit, s.headroom) == (500000, 1000000, 500000)
    assert s.headroom_frac == 0.5 and s.source == "cgroup.memory"


def test_cgroup_unlimited_falls_back_to_psutil(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    (tmp_path / "memory.current").write_text("1\n")
    s = CpuProbe(cgroup_root=tmp_path).snapshot()
    assert s.source == "psutil.virtual_memory" and s.limit > 0


def test_budget_probe_uses_rss():
    s = BudgetProbe(CpuProbe(), 64 * 2**40).snapshot()
    assert s.limit == 64 * 2**40 and 0 < s.used < s.limit


def test_headroom_never_negative():
    p = FakeProbe(limit=100, used=150)
    assert p.snapshot().headroom == 0
    assert MemSnapshot(1, 0, 0).headroom_frac == 0.0


def test_cgroup_cpu_quota(tmp_path):
    (tmp_path / "cpu.max").write_text("250000 100000\n")
    assert cgroup_cpu_limit(tmp_path) == 2.5
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cgroup_cpu_limit(tmp_path) is None


def test_settings_reject_nonsense():
    import pytest
    from pydantic import ValidationError

    from localhost_ai.config import Settings

    with pytest.raises(ValidationError):
        Settings(n_min=0)
    with pytest.raises(ValidationError):
        Settings(mem_reserve=1.5)


class FakeMx:
    """The four mlx.core calls MlxProbe makes."""

    def __init__(self, active, cache, max_ws):
        self.active, self.cache, self.max_ws = active, cache, max_ws

    def device_info(self):
        return {"max_recommended_working_set_size": self.max_ws}

    def get_active_memory(self):
        return self.active

    def get_cache_memory(self):
        return self.cache


def test_mlx_probe_limit_is_recommended_working_set_when_os_has_room(monkeypatch):
    from types import SimpleNamespace

    import localhost_ai.memory as memory

    monkeypatch.setattr(memory.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=8 * 2**30))
    s = memory.MlxProbe(FakeMx(active=2 * 2**30, cache=2**30, max_ws=10 * 2**30)).snapshot()
    assert (s.used, s.limit, s.headroom) == (2 * 2**30, 10 * 2**30, 8 * 2**30)
    assert s.source.startswith("mlx(")


def test_mlx_probe_limit_shrinks_with_os_available(monkeypatch):
    from types import SimpleNamespace

    import localhost_ai.memory as memory

    # other processes hold most of RAM: the limit is what we hold, our reusable cached
    # buffers, and what the OS could still hand out
    monkeypatch.setattr(memory.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=2**30))
    s = memory.MlxProbe(FakeMx(active=3 * 2**30, cache=2**29, max_ws=10 * 2**30)).snapshot()
    assert s.limit == 3 * 2**30 + 2**30 + 2**29
    assert s.headroom == 2**30 + 2**29


def test_probe_for_picks_mlx_by_backend(monkeypatch):
    import localhost_ai.memory as memory

    monkeypatch.setattr(memory, "MlxProbe", lambda: FakeProbe(limit=1, used=0))
    assert memory.probe_for("mps", backend="mlx").name == "fake"
    assert isinstance(memory.probe_for("cpu"), CpuProbe)
    wrapped = memory.probe_for("mps", budget=2**30, backend="mlx")
    assert isinstance(wrapped, BudgetProbe) and wrapped.name == "fake+budget"
