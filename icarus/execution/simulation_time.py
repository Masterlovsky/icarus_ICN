# This file is for a global simulation time, which is not used in the current

import threading
from collections import deque
from typing import Deque, DefaultDict
from collections import defaultdict
import time


class Sim_T(object):
    sim_time = threading.local()

    @staticmethod
    def get_sim_time():
        if not hasattr(Sim_T.sim_time, "current_time"):
            Sim_T.sim_time.current_time = 0
        return Sim_T.sim_time.current_time

    @staticmethod
    def set_sim_time(t):
        Sim_T.sim_time.current_time = t


class TimeWindowFrequencyTracker(object):
    def __init__(self, window_size: float = 60):
        # window_size. The size of the time window, based on the same time unit.
        self.window_size = window_size
        self.timestamps: DefaultDict[int, Deque[float]] = defaultdict(deque)

    def add_timestamp(self, v: int, timestamp: float):
        """Add a new timestamp to the record of node v."""
        # Remove outdated timestamps
        while self.timestamps[v] and timestamp - self.timestamps[v][0] > self.window_size:
            self.timestamps[v].popleft()

        # 添加新的时间戳
        self.timestamps[v].append(timestamp)

    def get_instantaneous_frequency(self, v: int, current_time: float) -> float:
        """Calculate the instantaneous frequency of node v."""
        while self.timestamps[v] and current_time - self.timestamps[v][0] > self.window_size:
            self.timestamps[v].popleft()

        # 计算窗口内的事件数量
        return len(self.timestamps[v]) / self.window_size


if __name__ == '__main__':
    tracker = TimeWindowFrequencyTracker(window_size=3)
    for i in range(5):
        tracker.add_timestamp(v=1, timestamp=time.time())
        time.sleep(1)
    freq = tracker.get_instantaneous_frequency(v=1, current_time=time.time())
    print(freq)
