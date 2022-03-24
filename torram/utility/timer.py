import collections
import logging
import time


class Timer:

    def __init__(self):
        self._logs = collections.defaultdict(list)
        self._tic = time.perf_counter()

    def log_dt(self, name: str):
        toc = time.perf_counter()
        dt = toc - self._tic
        self._logs[name].append(dt)
        self._tic = time.perf_counter()
        logging.debug(f"Logged runtime of {name} = {dt:.5f} s")

    def get_and_reset_logs(self):
        logs = self._logs.copy()
        self._logs = collections.defaultdict(list)
        return logs
