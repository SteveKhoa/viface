import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from lib.biocryp.binarizers import BaseBinarizer
from lib.biocryp.keygen import (
    BaseKeygen,
)
from benchmark import (
    utils_benchmark_report,
    utils_benchmark_runner,
    utils_data_transform,
    utils_data_load,
)


class BenchmarkCase:
    def __init__(
        self,
        binarizer: BaseBinarizer,
        keygen: BaseKeygen,
        turn_on_keygen_flag: bool,
        n_samples: int,
        benchmark_transform: utils_data_transform.DataTransform,
        data_loader: utils_data_load.DataLoad,
        binarizer_benchmark: utils_benchmark_runner.BaseBenchmarkRunner,
        keygen_benchmark: utils_benchmark_runner.BaseBenchmarkRunner,
        binarizer_report: utils_benchmark_report.BinarizerReport,
        keygen_report: utils_benchmark_report.KeygenReport,

    ):
        self.binarizer = binarizer
        self.keygen = keygen
        self.n_samples = n_samples
        self.benchmark_transform = benchmark_transform
        self.data_loader = data_loader

        self.benchmark_binarizer = binarizer_benchmark
        self.benchmark_keygen = keygen_benchmark

        self.binarizer_report = binarizer_report
        self.keygen_report = keygen_report

        # Flag to turn on/off key generation benchmarking.
        # This could help to improve benchmarking time.
        self.turn_on_keygen_flag = turn_on_keygen_flag

    def _get_result_and_report(self):
        result_benchmark_binarizer = self.benchmark_binarizer.get_result()
        result_benchmark_keygen = self.benchmark_keygen.get_result()

        self.binarizer_report.report(
            self.binarizer,
            self.n_samples,
            *result_benchmark_binarizer,
        )

        if self.turn_on_keygen_flag:
            self.keygen_report.report(
                self.n_samples,
                *result_benchmark_keygen,
            )

        print("BaseBenchmarkCase: done reporting.")

    def execute(self):
        people_struct_pair_list = self.benchmark_transform.transform()

        print("BaseBenchmarkCase: transforming dataset, done.")

        for _, people_struct_pair in enumerate(people_struct_pair_list):
            # Anonymous variable `x`, we simply don't care the structure of returned value
            # Just pass it on.
            x = self.data_loader.load_one_sample(people_struct_pair)

            y = self.benchmark_binarizer.run(*x)

            if self.turn_on_keygen_flag:
                self.benchmark_keygen.run(*y)

        print("BaseBenchmarkCase: done running.")

        self._get_result_and_report()