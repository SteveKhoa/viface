import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
from lib.biocryp.binarizers import BaseBinarizer
from benchmark import (
    utils_time_measurement,
    utils_benchmark_runner,
)
import time


class BenchmarkRunnerBinarizer(utils_benchmark_runner.BaseBenchmarkRunner):
    def __init__(self, binarizer: BaseBinarizer):
        super().__init__()

        # For distance analysis
        self.euclid_dist = []
        self.hamming_dist = []
        self.labels = []

        self.nsamples = 0.000000001  # not 0, so that no division by zero problem
        self.execution_time = 0.0

        self.binarizer = binarizer

    def get_result(self):
        # For reports
        self.euclid_dist = np.array(self.euclid_dist)
        self.hamming_dist = np.array(self.hamming_dist)
        self.labels = np.array(self.labels)
        self.execution_time /= self.nsamples

        return (self.euclid_dist, self.hamming_dist, self.labels, self.execution_time)

    @utils_time_measurement.decorator_time_measurement_log("BenchmarkRunnerBinarizer: one sample, done.")
    def run(self, base_signup_np_emb, base_login_np_emb, false_login_np_emb):
        self.nsamples += 1

        start = time.time()
        base_signup_np_boolarr = self.binarizer.binarise_asbool(base_signup_np_emb)
        end = time.time()
        self.execution_time += end - start

        base_login_np_boolarr = self.binarizer.binarise_asbool(base_login_np_emb)

        # Evaluate metrics
        individual_euclid_dist = np.linalg.norm(base_signup_np_emb - base_login_np_emb)
        individual_hamming_dist = np.count_nonzero(
            base_signup_np_boolarr != base_login_np_boolarr
        )

        # Record reference identity metrics
        self.euclid_dist += [individual_euclid_dist]
        self.hamming_dist += [individual_hamming_dist]
        self.labels += [True]

        false_login_np_boolarr = self.binarizer.binarise_asbool(false_login_np_emb)

        # Evaluate metrics
        individual_euclid_dist = np.linalg.norm(base_signup_np_emb - false_login_np_emb)
        individual_hamming_dist = np.count_nonzero(
            base_signup_np_boolarr != false_login_np_boolarr
        )

        # Record false identity metrics
        self.euclid_dist += [individual_euclid_dist]
        self.hamming_dist += [individual_hamming_dist]
        self.labels += [False]

        # Return for subsequent phases
        return (
            base_signup_np_boolarr,
            base_login_np_boolarr,
            false_login_np_boolarr
        )


class BenchmarkRunnerBinarizerForMaskThenLock(BenchmarkRunnerBinarizer):
    def __init__(self, binarizer: BaseBinarizer, registration_index: int = 0):
        super().__init__(binarizer)
        self.registration_index = registration_index
        self.binarizer = binarizer

    @utils_time_measurement.decorator_time_measurement_log("BenchmarkRunnerNaiveBinarizer: one sample, done.")
    def run(self, base_signup_np_embs, base_login_np_emb, false_login_np_emb):
        self.nsamples += 1

        base_signup_np_boolarrs = [self.binarizer.binarise_asbool(base_signup_np_emb) for base_signup_np_emb in base_signup_np_embs]
        base_login_np_boolarr = self.binarizer.binarise_asbool(base_login_np_emb)

        # Evaluate metrics
        individual_euclid_dist = np.linalg.norm(
            base_signup_np_embs[self.registration_index] - base_login_np_emb
        )
        individual_hamming_dist = np.count_nonzero(
            base_signup_np_boolarrs[self.registration_index] != base_login_np_boolarr
        )

        # Record reference identity metrics
        self.euclid_dist += [individual_euclid_dist]
        self.hamming_dist += [individual_hamming_dist]
        self.labels += [True]

        false_login_np_boolarr = self.binarizer.binarise_asbool(false_login_np_emb)

        # Evaluate metrics
        individual_euclid_dist = np.linalg.norm(
            base_signup_np_embs[self.registration_index] - false_login_np_emb
        )
        individual_hamming_dist = np.count_nonzero(
            base_signup_np_boolarrs[self.registration_index] != false_login_np_boolarr
        )

        # Record false identity metrics
        self.euclid_dist += [individual_euclid_dist]
        self.hamming_dist += [individual_hamming_dist]
        self.labels += [False]

        return (
            base_signup_np_boolarrs, 
            base_login_np_boolarr, 
            false_login_np_boolarr,
        )