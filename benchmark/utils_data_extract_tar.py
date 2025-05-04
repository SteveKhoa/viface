from benchmark import utils_time_measurement
import tarfile
import io
import numpy as np


class DataExtractTarToFileDictionary(object):
    def __init__(self, path: str, load_fn = np.load):
        self.datapath = path
        self.load_fn = load_fn

    @utils_time_measurement.decorator_time_measurement_log("TarToFileDictionary: extract tar file to file dictionary")
    def extract(self) -> dict:
        """Extract tar file to a dictionary, where keys are individual file paths originally."""

        file_dictionary = {}
        with tarfile.open(self.datapath) as tar:
            for member in tar.getmembers():
                buff = io.BytesIO()
                buff.write(tar.extractfile(member).read())
                buff.seek(0)
                file_dictionary[f"{member.name}"] = self.load_fn(buff)
        return file_dictionary