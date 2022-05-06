import collections
import logging
import time


class Timer:
    """Tiny profiling class for timings from explicit code calls.

    >>> timer = Timer()
    >>> for _ in range(10):
    >>>     x = 1
    >>>     timer.log_dt("assigned x")
    >>>     y = 1
    >>>     timer.log_dt("assigned y")
    >>> logs = timer.get_and_reset_logs()
    """

    def __init__(self):
        self._logs = collections.defaultdict(list)
        self._tic = time.perf_counter()

    def log_dt(self, name: str):
        toc = time.perf_counter()
        dt = toc - self._tic
        self._logs[name].append(dt)
        self.reset()
        logging.debug(f"Logged runtime of {name} = {dt:.5f} s")

    def get_and_reset_logs(self):
        logs = self._logs.copy()
        self._logs = collections.defaultdict(list)
        return logs

    def reset(self):
        self._tic = time.perf_counter()
