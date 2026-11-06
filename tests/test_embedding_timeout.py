import time
import pytest
from pipeline.planner_novel import _time_limit, EmbeddingTimeout


def test_time_limit_aborts_slow_block():
    t0 = time.time()
    with pytest.raises(EmbeddingTimeout):
        with _time_limit(1):
            time.sleep(5)
    assert time.time() - t0 < 3  # interrupted well before the 5s sleep finished


def test_time_limit_allows_fast_block():
    with _time_limit(5):
        time.sleep(0.01)  # completes normally, no exception


def test_time_limit_disabled_is_noop():
    with _time_limit(0):
        time.sleep(0.01)  # 0 disables the guard; must not raise
