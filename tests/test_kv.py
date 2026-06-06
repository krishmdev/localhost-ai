import torch

from localhost_ai.engine import kv


def cache_for(rows, heads=2, dim=3, layers=2):
    """rows: list of real lengths. Content encodes (row, position) so we can check it moved
    correctly: value = 100*row_id + pos + 1, padding = 0."""
    t = max(rows)
    mask = torch.zeros(len(rows), t, dtype=torch.long)
    k = torch.zeros(len(rows), heads, t, dim)
    for i, n in enumerate(rows):
        mask[i, t - n:] = 1
        for p in range(n):
            k[i, :, t - n + p, :] = 100 * (i + 1) + p + 1
    return kv.build([(k.clone(), -k.clone()) for _ in range(layers)]), mask


def real(cache, mask, row):
    keys = kv.layers(cache)[0][0][row, 0, :, 0]
    return keys[mask[row].bool()].tolist()


def test_merge_left_pads_shorter_side():
    a, ma = cache_for([2, 4])
    b, mb = cache_for([6])
    c, m = kv.merge(a, ma, b, mb)
    assert m.shape == (3, 6)
    assert kv.seq_len(c) == 6
    assert m.sum(1).tolist() == [2, 4, 6]
    assert real(c, m, 0) == [101, 102]
    assert real(c, m, 1) == [201, 202, 203, 204]
    assert real(c, m, 2) == [101, 102, 103, 104, 105, 106]  # b's row 0
    for k, v in kv.layers(c):
        assert torch.equal(v, -k)
        assert torch.all(k[~m.bool().unsqueeze(1).unsqueeze(-1).expand_as(k)] == 0)


def test_select_rows_trims_shared_padding():
    c, m = cache_for([2, 6, 3])
    c2, m2 = kv.select_rows(c, m, [0, 2])
    assert m2.shape == (2, 3)  # the long row left; 3 all-padding columns dropped
    assert real(c2, m2, 0) == [101, 102]
    assert real(c2, m2, 1) == [301, 302, 303]
    assert kv.seq_len(c2) == 3


def test_crop_undoes_partial_step():
    c, m = cache_for([3, 3])
    layers = kv.layers(c)
    c.layers[0].keys = torch.cat([layers[0][0], torch.ones(2, 2, 1, 3)], dim=2)
    c.layers[0].values = torch.cat([layers[0][1], torch.ones(2, 2, 1, 3)], dim=2)
    kv.crop(c, 3)
    assert all(k.shape[2] == 3 for k, _ in kv.layers(c))


def test_positions():
    m = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert kv.prefill_positions(m).tolist() == [[0, 0, 0, 1], [0, 1, 2, 3]]
    assert kv.next_positions(m).tolist() == [[2], [4]]


def test_nbytes():
    c, _ = cache_for([4, 4])
    # 2 layers * (K+V) * 2 rows * 2 heads * 4 pos * 3 dim * 4 bytes
    assert kv.nbytes(c) == 2 * 2 * 2 * 2 * 4 * 3 * 4
