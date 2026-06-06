from localhost_ai.engine.detok import IncrementalDetokenizer


class ByteTok:
    """Tokens are UTF-8 bytes, like BPE byte fallback: a multibyte char spans tokens."""

    def decode(self, ids, skip_special_tokens=True):
        return bytes(ids).decode("utf-8", errors="replace")


def ids(s):
    return list(s.encode("utf-8"))


def run(text, stop=()):
    d = IncrementalDetokenizer(ByteTok(), stop)
    out = []
    for t in ids(text):
        out.append(d.push(t))
        if d.stopped:
            break
    out.append(d.flush())
    return "".join(out), out, d


def test_multibyte_chars_are_not_split():
    text, pieces, _ = run("héllo → wörld")
    assert text == "héllo → wörld"
    assert all("�" not in p for p in pieces)


def test_stop_string_cut_and_hold_back():
    text, pieces, d = run("abc STOP def", stop=("STOP",))
    assert text == "abc " and d.stopped
    assert "S" not in "".join(pieces)


def test_partial_stop_prefix_released_on_flush():
    text, _, d = run("abc ST", stop=("STOP",))
    assert text == "abc ST" and not d.stopped


def test_no_stop():
    text, _, _ = run("plain text")
    assert text == "plain text"
