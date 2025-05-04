import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import abc

class BaseBenchmarkRunner(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def run(self):
        """Stateful runs."""
        pass

    @abc.abstractmethod
    def get_result(self):
        """Get result from `run`.
        In case `run` is executed many times, the result could be accumulated from multiple runs.

        Therefore, it is necessary that this function implements that accumulated result."""
        pass