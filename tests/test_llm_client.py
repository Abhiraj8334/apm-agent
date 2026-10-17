import pytest

from src.agent.llm_client import is_retryable, retry_with_backoff


class FakeAPIError(Exception):
    """Mimics google.genai errors, which expose an HTTP status as `.code`."""

    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


def test_retries_transient_error_then_succeeds():
    calls = {"n": 0}
    sleeps: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FakeAPIError(503)  # "high demand"
        return "ok"

    result = retry_with_backoff(flaky, max_attempts=5, sleep=sleeps.append)

    assert result == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2
    assert sleeps[1] > sleeps[0]  # backoff grows


def test_does_not_retry_permanent_errors():
    calls = {"n": 0}

    def bad_key():
        calls["n"] += 1
        raise FakeAPIError(401)

    with pytest.raises(FakeAPIError):
        retry_with_backoff(bad_key, max_attempts=5, sleep=lambda s: None)

    assert calls["n"] == 1  # gave up immediately, no pointless retries


def test_gives_up_after_max_attempts():
    calls = {"n": 0}

    def always_busy():
        calls["n"] += 1
        raise FakeAPIError(503)

    with pytest.raises(FakeAPIError):
        retry_with_backoff(always_busy, max_attempts=3, sleep=lambda s: None)

    assert calls["n"] == 3


def test_is_retryable_classification():
    assert is_retryable(FakeAPIError(429))
    assert is_retryable(FakeAPIError(503))
    assert not is_retryable(FakeAPIError(400))
    assert not is_retryable(FakeAPIError(401))
    assert not is_retryable(ValueError("no code attribute"))
