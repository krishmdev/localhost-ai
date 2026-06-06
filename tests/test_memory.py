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
