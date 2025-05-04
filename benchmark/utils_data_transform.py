import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import abc


class DataTransform:
    @abc.abstractmethod
    def transform(self) -> list:
        pass