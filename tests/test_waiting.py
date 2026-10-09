import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import waiting  # noqa: E402


class FakeBox:
    def __init__(self):
        self.shown, self.cleared = [], False

    def markdown(self, html, unsafe_allow_html=False):
        self.shown.append(html)

    def empty(self):
        self.cleared = True


class FakeSt:
    def __init__(self, broken=False):
        self.box, self.broken, self.spinner_text = FakeBox(), broken, None

    def empty(self):
        if self.broken:
            raise RuntimeError("no script context")
        return self.box

    def spinner(self, text):
        self.spinner_text = text
        from contextlib import nullcontext
        return nullcontext()


def slow(secret, delay=0.35):
    time.sleep(delay)
    return f"done {len(secret)}"


def test_words_rotate_while_the_call_runs_and_it_runs_once(monkeypatch):
    fake = FakeSt()
    monkeypatch.setattr(waiting, "st", fake)
    calls = []
    out = waiting.run_with_status(lambda k: (calls.append(1), slow(k))[1], "sk-secret", interval=0.1)
    assert out == "done 9" and calls == [1]
    words_seen = [next(w for w in waiting.STATUS_WORDS if w in html) for html in fake.box.shown]
    assert len(set(words_seen)) >= 2 and words_seen[0] == waiting.STATUS_WORDS[0]
    assert fake.box.cleared
    assert all("sk-secret" not in html for html in fake.box.shown)
    assert "shimmer" in fake.box.shown[0]


def test_falls_back_to_plain_spinner_without_running_twice(monkeypatch):
    fake = FakeSt(broken=True)
    monkeypatch.setattr(waiting, "st", fake)
    calls = []
    assert waiting.run_with_status(lambda: calls.append(1) or "ok", fallback_text="Working") == "ok"
    assert calls == [1] and fake.spinner_text == "Working"


def test_exception_in_call_reaches_the_caller(monkeypatch):
    monkeypatch.setattr(waiting, "st", FakeSt())

    def boom():
        raise ValueError("bad")

    with pytest.raises(ValueError):
        waiting.run_with_status(boom, interval=0.05)


def test_animation_failure_still_returns_the_result(monkeypatch):
    fake = FakeSt()

    def bad_markdown(html, unsafe_allow_html=False):
        raise RuntimeError("ui gone")

    fake.box.markdown = bad_markdown
    monkeypatch.setattr(waiting, "st", fake)
    assert waiting.run_with_status(slow, "abc", interval=0.05) == "done 3"


def test_default_interval_and_words():
    assert waiting.INTERVAL_SECONDS == 1.5
    for w in ("Circulating", "Fishing for facts", "Sweeping the hole"):
        assert w in waiting.STATUS_WORDS
