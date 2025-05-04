import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
from lib.biocryp.keygen import (
    BaseKeygen,
    MaskThenLockFuzzyExtractor,
)
from lib.biocryp.contexts import BitString
from benchmark import (
    utils,
    utils_time_measurement,
    utils_benchmark_runner,
)
import time


class BenchmarRunnerkKeygen(utils_benchmark_runner.BaseBenchmarkRunner):
    def __init__(self, keygen: BaseKeygen):
        super().__init__()
        self.keygen = keygen

        self.encrypted_total_false_notmatch = 0
        self.encrypted_total_false_match = 0

        self.nsamples = 0.000000001  # not 0, so that no division by zero problem

        self.total_registration_time = 0
        self.total_login_time = 0

    def get_result(self):
        self.encrypted_total_false_notmatch /= float(self.nsamples)
        self.encrypted_total_false_match /= float(self.nsamples)

        self.total_registration_time /= float(self.nsamples)
        self.total_login_time /= float(self.nsamples)

        return (
            self.encrypted_total_false_match, 
            self.encrypted_total_false_notmatch,
            self.total_registration_time,
            self.total_login_time,
        )

    @utils_time_measurement.decorator_time_measurement_log("BenchmarkKeygen: one sample, done.")
    def run(self, base_signup_np_boolarr, base_login_np_boolarr, false_login_np_boolarr):
        self.nsamples += 1

        signup_raw = BitString(np.array(base_signup_np_boolarr, dtype=np.uint8))
        login_raw = BitString(np.array(base_login_np_boolarr, dtype=np.uint8))
        false_login_raw = BitString(np.array(false_login_np_boolarr, dtype=np.uint8))

        start = time.time()
        signup_key, keyseed = self.keygen.generate(signup_raw.as_bytes())
        end = time.time()
        self.total_registration_time += (end - start)

        start = time.time()
        login_key, _ = self.keygen.generate(login_raw.as_bytes(), keyseed)
        end = time.time()
        self.total_login_time += (end - start)

        # Skip benchmarking this function (already benchmarked above)
        false_login_key, _ = self.keygen.generate(false_login_raw.as_bytes(), keyseed)

        self.encrypted_total_false_notmatch += int(not (login_key == signup_key))
        self.encrypted_total_false_match += int(false_login_key == signup_key)


class BenchmarkRunnerMaskThenLockKeyGen(BenchmarRunnerkKeygen):
    def __init__(self, keygen: MaskThenLockFuzzyExtractor):
        super().__init__(keygen)

    @utils_time_measurement.decorator_time_measurement_log("BenchmarkKeygen: one sample, done.")
    def run(self, base_signup_np_boolarrs, base_login_np_boolarr, false_login_np_boolarr):
        self.nsamples += 1

        signup_raws = [BitString(np.array(x, dtype=np.uint8)) for x in base_signup_np_boolarrs]
        login_raw = BitString(np.array(base_login_np_boolarr, dtype=np.uint8))
        false_login_raw = BitString(np.array(false_login_np_boolarr, dtype=np.uint8))

        start = time.time()
        signup_key, keyseed = self.keygen.generate(b"".join([x.as_bytes() for x in signup_raws]))
        end = time.time()
        self.total_registration_time += (end - start)

        start = time.time()
        login_key, _ = self.keygen.generate(login_raw.as_bytes(), keyseed)
        end = time.time()
        self.total_login_time += (end - start)

        # Skip benchmarking this function (already did above)
        false_login_key, _ = self.keygen.generate(false_login_raw.as_bytes(), keyseed)

        self.encrypted_total_false_notmatch += int(not (login_key == signup_key))
        self.encrypted_total_false_match += int(false_login_key == signup_key)