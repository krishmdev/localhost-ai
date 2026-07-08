"""PrefixCache bookkeeping, backend-free: lookup, when a new prefix is proposed, LRU eviction
under the entry and byte budgets, and not rebuilding a prefix too big for the budget."""

import torch

from localhost_ai.engine.prefix import PrefixCache, common_prefix, prefill_with_prefixes

SYS = list(range(100, 140))  # 40 shared tokens


def test_common_prefix():
    assert common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix([], [1]) == 0
    assert common_prefix([1, 2], [1, 2]) == 2


def test_propose_needs_a_recent_prompt_sharing_enough_tokens():
    pc = PrefixCache(budget_bytes=1000, min_tokens=32)
    assert pc.propose(SYS + [1, 2]) == 0  # nothing recent to compare with
    assert pc.propose(SYS + [3]) == 40
    assert pc.propose(list(range(20)) + [5]) == 0  # shares nothing
    pc2 = PrefixCache(budget_bytes=1000, min_tokens=32)
    pc2.propose(SYS[:20] + [1])
    assert pc2.propose(SYS[:20] + [2]) == 0  # 20 < min_tokens


def test_propose_leaves_a_token_to_run_and_skips_small_gains():
    pc = PrefixCache(budget_bytes=1000, min_tokens=8)
    pc.propose(SYS)
    assert pc.propose(SYS) == 39  # identical prompt: all but its last token
    pc.propose(SYS + [1, 2, 3])
    assert pc.propose(SYS + [1, 2, 4], have=40) == 0  # 42 is not worth a second entry
    pc.propose(SYS + list(range(60)))
    assert pc.propose(SYS + list(range(60)) + [7], have=40) == 100


def test_lookup_takes_the_longest_stored_prefix():
    pc = PrefixCache(budget_bytes=1000)
    pc.add(SYS[:10], "short", 10)
    pc.add(SYS, "long", 10)
    assert pc.lookup(SYS + [1]).state == "long"
    assert pc.lookup(SYS[:20]).state == "short"
    assert pc.lookup(SYS).state == "short"  # the 40-token entry would leave nothing to run
    assert pc.lookup([1, 2, 3]) is None


def test_lru_eviction_by_count_and_bytes():
    pc = PrefixCache(budget_bytes=100, max_entries=2)
    a = pc.add([1] * 5, "a", 40)
    pc.add([2] * 5, "b", 40)
    pc.touch(a)  # a is now the most recent
    pc.add([3] * 5, "c", 40)  # over max_entries: b goes
    assert [e.state for e in pc.entries.values()] == ["a", "c"]
    pc.add([4] * 5, "d", 90)  # over the byte budget: a and c go
    assert [e.state for e in pc.entries.values()] == ["d"] and pc.evictions == 3
    assert pc.add([5] * 5, "e", 101) is None  # can never fit
    assert pc.clear() == 90 and pc.nbytes == 0


def test_stats():
    pc = PrefixCache(budget_bytes=100)
    e = pc.add([1] * 5, "a", 40)
    pc.touch(e)
    st = pc.stats()
    assert st["prefix_entries"] == 1 and st["prefix_bytes"] == 40
    assert st["prefix_hits"] == 1 and st["prefix_hit_tokens"] == 5


def test_a_prefix_over_the_budget_is_built_once_then_not_proposed_again():
    class Runner:
        builds = 0

        def _build_prefix(self, tokens):
            Runner.builds += 1
            return "state", 10 * len(tokens)  # 400 bytes for 40 tokens

        def _prefill(self, seqs, entry):
            return "batch", torch.zeros(len(seqs), 4)

    pc = PrefixCache(budget_bytes=100, min_tokens=8)
    system = list(range(100, 140))
    for i in range(6):
        prefill_with_prefixes(Runner(), pc, [system + [i, i + 1]])
    assert Runner.builds == 1 and not pc.entries and pc.too_big == 40
    # a shorter shared prefix, which may fit, is still proposed
    assert pc.propose(system[:8] + [7, 7]) == 8
