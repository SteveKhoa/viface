import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import sklearn.preprocessing
from commons import Folder
from collections import defaultdict
import numpy as np
import random
from lib.biocryp.binarizers import BaseBinarizer, DROBA, DROBASVC, Static
from lib.biocryp.keygen import (
    DefaultKeygen,
    BaseKeygen,
    YimingFuzzyExtractor,
    MaskThenLockFuzzyExtractor,
)
from lib.biocryp.contexts import BitString
import re
from random import randint
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import tarfile
import io
import abc
import sklearn


def fib_list(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]

    result = [0, 1]
    while len(result) < n:
        result.append(result[-1] + result[-2])
    return result


def to_bytestring(bool_arr, length):
    x = sum(map(lambda x: x[1] << x[0], enumerate(bool_arr)))
    bin_ = int(x).to_bytes(length=length, byteorder="big", signed=True)
    return bin_


def measure_time(description: str):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            retval = func(*args, **kwargs)
            end = time.time()
            print(description, ":", round(end - start, 4), "s")
            return retval

        return wrapper

    return decorator


# # Planned to delete soon, temporarily keep here for reference
#
# class PeopleFolder(Folder):
#     def __init__(self, folderpath: str = "./lfw.embeddings"):
#         super().__init__(folderpath)

#         # labels array is strictly ordered, do not change it
#         _, self.namemap, self.namecount = self._get_labelarr()

#     def _get_labelarr(self):
#         namemap = defaultdict(int)
#         namecount = defaultdict(int)
#         next_label = 0
#         labelarr = [-1] * len(self.files)

#         for i, file in enumerate(self.query(".*")):
#             name = "_".join(tuple(file.split("_")[:-1]))

#             if name not in namemap:
#                 namemap[name] = next_label
#                 next_label += 1
#             else:
#                 pass
#             namecount[name] += 1

#             labelarr[i] = namemap[name]

#         return labelarr, namemap, namecount

#     # def _core_query(self, selected_labels: List[int]):
#     #     mask = np.array([False] * self.nitems())
#     #     for label in selected_labels:
#     #         r = utilsbm.argfind(label, self.labels)
#     #         mask[r] = True

#     #     selected = self.query_byidx(mask)
#     #     return selected

#     def exclude(self, count: int = 1):
#         """
#         Exclude any names with count <= `count`.
#         """
#         included = dict(filter(lambda item: item[1] > count, self.namecount.items()))
#         self.namemap = dict(
#             filter(lambda item: item[0] in included, self.namemap.items())
#         )

#     def names(self):
#         return list(self.namemap.keys())

#     # def query_multisamples(self):
#     #     """
#     #     Query all names
#     #     """
#     #     count_dict = defaultdict(int)
#     #     for label in self.labels:
#     #         count_dict[label] += 1
#     #     included = filter(lambda item: item[1] > 1, count_dict.items())
#     #     labels = list(map(lambda item: item[0], included))

#     #     return self._core_query(labels)

#     # def query_common(self, topk: int = 5):
#     #     """
#     #     Query most common names, but not actually retrieve the data, yet.
#     #     """

#     #     # Scan & count the labels in the array
#     #     count_dict = defaultdict(int)
#     #     for label in self.labels:
#     #         count_dict[label] += 1

#     #     label_items = sorted(count_dict.items(), key=lambda item: item[1], reverse=True)
#     #     topk_labels = list(map(lambda x: x[0], label_items[:topk]))

#     #     mask = np.array([False] * self.nitems())
#     #     for label in topk_labels:
#     #         r = utilsbm.argfind(label, self.labels)
#     #         mask[r] = True

#     #     selected = self.query_byidx(mask)
#     #     return selected


class TarOnRam(object):
    def __init__(self, path: str):
        self.datapath = path

    @measure_time("Load tar file")
    def load(self) -> dict:
        """Load tar file to a dictionary, where keys are individual file paths originally."""

        virtual_fs = {}
        with tarfile.open(self.datapath) as tar:
            for member in tar.getmembers():
                buff = io.BytesIO()
                buff.write(tar.extractfile(member).read())
                buff.seek(0)
                virtual_fs[f"{member.name}"] = np.load(buff)
        return virtual_fs


class BenchmarkLoader(object):
    @abc.abstractmethod
    def load(self):
        """Load entire datasets into Memory"""
        pass


class PeopleLoader(BenchmarkLoader):
    def __init__(self, binarizer):
        super().__init__()
        self.binarizer: BaseBinarizer = binarizer

    def _group_and_filter(self, virtual_fs: dict):
        """This utility function group the file paths into distinct values (people names)
        and filter out people with only one sample (because one sample is not useful
        for downstream tasks).
        """
        keys = list(virtual_fs.keys())
        s = defaultdict(int)
        for key in keys:
            justname = "_".join(key.split("/")[-1].split(".")[0].split("_")[:-1])
            s[justname] += 1

        filtered_keys = dict(list(filter(lambda x: x[1] > 1, s.items()))).keys()
        return filtered_keys

    @measure_time("Load samples")
    def load(self, nsamples):
        backend = TarOnRam("./lfw.embeddings.tar")
        virtual_fs = backend.load()
        names = self._group_and_filter(virtual_fs)
        filepaths = list(virtual_fs.keys())

        all_people = []
        for i in range(nsamples):
            if i % 100 == 0:
                print(f"{i}\{nsamples}")

            sampled = random.sample(names, 2)

            # Append prefix and postfix with wildcard so that regex can match it
            sampled = [".*" + sample + ".*" for sample in sampled]

            # Preselect so that we don't have to run filter multiple times
            selected = list(
                filter(
                    lambda fpath: re.match(sampled[0], fpath)
                    or re.match(sampled[1], fpath),
                    filepaths,
                )
            )

            people = dict()
            for name in sampled:
                embeddings = [
                    virtual_fs[fname]
                    for fname in filter(lambda fpath: re.match(name, fpath), selected)
                ]

                # images = []  # see no immediate value right now, however I feel it is important in the future
                people[name] = {
                    # "images": images,  # see no immediate value right now, however I feel it is important in the future
                    "embeddings": sklearn.preprocessing.normalize(embeddings),
                    "n": len(embeddings),
                }

            # Record this people
            people_lst = list(people.items())
            all_people += [people_lst]

        return all_people

    def get_sample(self, people_lst):
        # Get a true identity and a false identity
        true_identity = 0
        false_identity = 1

        # Acquire base
        base_embeddings = people_lst[true_identity][-1]["embeddings"]
        base_np_boolarrs = [
            self.binarizer.binarise_asbool(embedding) for embedding in base_embeddings
        ]
        base_length = people_lst[true_identity][-1]["n"]

        # Acquire false
        false_embeddings = people_lst[false_identity][-1]["embeddings"]
        false_np_boolarrs = [
            self.binarizer.binarise_asbool(embedding) for embedding in false_embeddings
        ]
        false_length = people_lst[false_identity][-1]["n"]

        return (
            (base_embeddings, base_np_boolarrs, base_length),
            (false_embeddings, false_np_boolarrs, false_length),
        )


class PopularPeopleLoader(PeopleLoader):
    def _group_and_filter(self, virtual_fs: dict):
        """This utility function group the file paths into distinct values (people names)
        and filter out people with only one sample (because one sample is not useful
        for downstream tasks).
        """
        keys = list(virtual_fs.keys())
        s = defaultdict(int)
        for key in keys:
            justname = "_".join(key.split("/")[-1].split(".")[0].split("_")[:-1])
            s[justname] += 1

        filtered_keys = dict(list(filter(lambda x: x[1] > 10, s.items()))).keys()
        print("number of people after filtered", len(filtered_keys))
        return filtered_keys


class PairLoader(PeopleLoader):
    def __init__(self, binarizer):
        super().__init__(binarizer)

    def get_sample(self, people_lst):
        (
            (base_embeddings, base_np_boolarrs, base_length),
            (false_embeddings, false_np_boolarrs, false_length),
        ) = super().get_sample(people_lst)

        # Select two random samples in reference (base) identity
        base_select_signup = randint(0, base_length - 1)
        base_select_login = randint(0, base_length - 1)
        while (
            base_select_login == base_select_signup
        ):  # make sure two samples are not the same
            base_select_login = randint(0, base_length - 1)

        base_signup_np_boolarr = base_np_boolarrs[base_select_signup]
        base_login_np_boolarr = base_np_boolarrs[base_select_login]
        base_signup_np_emb = base_embeddings[base_select_signup]
        base_login_np_emb = base_embeddings[base_select_login]

        # Select one random sample from false identity
        false_select_login = randint(0, false_length - 1)
        false_login_np_boolarr = false_np_boolarrs[false_select_login]
        false_login_np_emb = false_embeddings[false_select_login]

        return (
            (
                base_signup_np_boolarr,
                base_login_np_boolarr,
                base_signup_np_emb,
                base_login_np_emb,
            ),
            (
                false_login_np_boolarr,
                false_login_np_emb,
            ),
        )


class PairLoaderMany(PopularPeopleLoader):
    """Designed specifically for my fuzzy extractor.
    This loads many samples of the same base identity.
    """

    def __init__(self, binarizer: MaskThenLockFuzzyExtractor):
        super().__init__(binarizer)

    def get_sample(self, people_lst):
        (
            (base_embeddings, base_np_boolarrs, base_length),
            (false_embeddings, false_np_boolarrs, false_length),
        ) = super().get_sample(people_lst)

        # Select two random samples in reference (base) identity
        base_select_signup = randint(0, base_length - 1)
        base_select_login = randint(0, base_length - 1)
        while (
            base_select_login == base_select_signup
        ):  # make sure two samples are not the same
            base_select_login = randint(0, base_length - 1)

        base_login_np_boolarr = base_np_boolarrs[base_select_login]
        base_login_np_emb = base_embeddings[base_select_login]

        # A bit different with traditional PairLoader, in this context,
        # the loader will loads many boolarrs and embs as registration.
        #
        # All the signup samples must different from the login sample
        #
        # It is required by our dedicated fuzzy extractor.
        base_np_boolarrs = np.array(
            base_np_boolarrs
        )  # so that slicing operation could work
        base_embeddings = np.array(base_embeddings)
        if base_select_login < base_length - 1:
            base_signup_np_boolarrs = np.concatenate(
                [
                    base_np_boolarrs[:base_select_login],
                    base_np_boolarrs[base_select_login + 1 :],
                ]
            )
            base_signup_np_embs = np.concatenate(
                [
                    base_embeddings[:base_select_login],
                    base_embeddings[base_select_login + 1 :],
                ]
            )
        else:
            base_signup_np_boolarrs = base_np_boolarrs[:base_select_login]
            base_signup_np_embs = base_embeddings[:base_select_login]

        # Select one random sample from false identity
        false_select_login = randint(0, false_length - 1)
        false_login_np_boolarr = false_np_boolarrs[false_select_login]
        false_login_np_emb = false_embeddings[false_select_login]

        return (
            (
                base_signup_np_boolarrs,
                base_login_np_boolarr,
                base_signup_np_embs,
                base_login_np_emb,
            ),
            (
                false_login_np_boolarr,
                false_login_np_emb,
            ),
        )


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


class BenchmarkBinarizer(BaseBenchmarkRunner):
    def __init__(
        self,
        pair_loader: PairLoader,
    ):
        super().__init__()

        # For distance analysis
        self.euclid_dist = []
        self.hamming_dist = []
        self.labels = []

        # Loader as composition
        self.pair_loader = pair_loader

        self.nsamples = 0.000000001  # not 0, so that no division by zero problem

    def get_result(self):
        # For reports
        self.euclid_dist = np.array(self.euclid_dist)
        self.hamming_dist = np.array(self.hamming_dist)
        self.labels = np.array(self.labels)

        return (self.euclid_dist, self.hamming_dist, self.labels)

    @measure_time("Benchmarked binarizer, one sample")
    def run(self, people_lst=None):
        self.nsamples += 1
        (
            (
                base_signup_np_boolarr,
                base_login_np_boolarr,
                base_signup_np_emb,
                base_login_np_emb,
            ),
            (
                false_login_np_boolarr,
                false_login_np_emb,
            ),
        ) = self.pair_loader.get_sample(people_lst)

        # Evaluate metrics
        individual_euclid_dist = np.linalg.norm(base_signup_np_emb - base_login_np_emb)
        individual_hamming_dist = np.count_nonzero(
            base_signup_np_boolarr != base_login_np_boolarr
        )

        # Record reference identity metrics
        self.euclid_dist += [individual_euclid_dist]
        self.hamming_dist += [individual_hamming_dist]
        self.labels += [True]

        # Evaluate metrics
        individual_euclid_dist = np.linalg.norm(base_signup_np_emb - false_login_np_emb)
        individual_hamming_dist = np.count_nonzero(
            base_signup_np_boolarr != false_login_np_boolarr
        )

        # Record false identity metrics
        self.euclid_dist += [individual_euclid_dist]
        self.hamming_dist += [individual_hamming_dist]
        self.labels += [False]


class BenchmarkFabulousBinarizer(BenchmarkBinarizer):
    def __init__(self, pair_loader: PairLoader, registration_index: int = 0):
        super().__init__(pair_loader)
        self.registration_index = registration_index

    @measure_time("Benchmarked binarizer, one sample")
    def run(self, people_lst=None):
        self.nsamples += 1
        (
            (
                base_signup_np_boolarrs,
                base_login_np_boolarr,
                base_signup_np_embs,
                base_login_np_emb,
            ),
            (
                false_login_np_boolarr,
                false_login_np_emb,
            ),
        ) = self.pair_loader.get_sample(people_lst)

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


class BenchmarkKeygen(BaseBenchmarkRunner):
    def __init__(self, keygen: BaseKeygen):
        super().__init__()
        self.keygen = keygen

        self.encrypted_total_false_notmatch = 0
        self.encrypted_total_false_match = 0

        self.nsamples = 0.000000001  # not 0, so that no division by zero problem

    def get_result(self):
        self.encrypted_total_false_notmatch /= float(self.nsamples)
        self.encrypted_total_false_match /= float(self.nsamples)

        return (self.encrypted_total_false_notmatch, self.encrypted_total_false_match)

    @measure_time("Benchmarked Keygen, one sample")
    def run(self, people_lst=None, pair_loader: PairLoader = None):
        self.nsamples += 1
        (
            (
                base_signup_np_boolarr,
                base_login_np_boolarr,
                _,
                _,
            ),
            (
                false_login_np_boolarr,
                _,
            ),
        ) = pair_loader.get_sample(people_lst)

        l_bytes = int(self.keygen.info().get("length_bitstring") / 8)
        signup_raw = to_bytestring(base_signup_np_boolarr, l_bytes)
        login_raw = to_bytestring(base_login_np_boolarr, l_bytes)
        false_login_raw = to_bytestring(false_login_np_boolarr, l_bytes)

        signup_key, keyseed = self.keygen.generate(signup_raw)
        login_key, _ = self.keygen.generate(login_raw, keyseed)
        false_login_key, _ = self.keygen.generate(false_login_raw, keyseed)

        self.encrypted_total_false_notmatch += int(not (login_key == signup_key))
        self.encrypted_total_false_match += int(false_login_key == signup_key)


class BenchmarkMyFabulousKeygen(BenchmarkKeygen):
    def __init__(self, keygen: BaseKeygen):
        super().__init__(keygen)

    @measure_time("Benchmarked Keygen, one sample")
    def run(self, people_lst=None, pair_loader: PairLoaderMany = None):
        self.nsamples += 1
        (
            (
                base_signup_np_boolarrs,
                base_login_np_boolarr,
                _,
                _,
            ),
            (
                false_login_np_boolarr,
                _,
            ),
        ) = pair_loader.get_sample(people_lst)

        login_raw = BitString(np.array(base_login_np_boolarr, dtype=np.uint8))
        false_login_raw = BitString(np.array(false_login_np_boolarr, dtype=np.uint8))

        signup_raws = [
            BitString(np.array(x, dtype=np.uint8)) for x in base_signup_np_boolarrs
        ]
        signup_raws = [x.as_bytes() for x in signup_raws]
        signup_raws = b"".join(signup_raws)
        signup_key, keyseed = self.keygen.generate(signup_raws)
        login_key, _ = self.keygen.generate(login_raw.as_bytes(), keyseed)
        false_login_key, _ = self.keygen.generate(false_login_raw.as_bytes(), keyseed)

        # print(len(signup_key))
        # print(len(login_key))
        # print(len(false_login_key))
        # print("\n\n\n")

        self.encrypted_total_false_notmatch += int(not (login_key == signup_key))
        self.encrypted_total_false_match += int(false_login_key == signup_key)


class BaseBenchmarkReport(object):
    @abc.abstractmethod
    def report(self):
        pass


class BinarizerReport(BaseBenchmarkReport):
    def __init__(self, benchmark_name: str):
        self.benchmark_name = f"{benchmark_name}"

    def compute_overlap_auc(self, distribution_01, distribution_02):
        """
        Compute the overlapping area between two distributions.

        Parameters:
        distribution_01 (array-like): First distribution data
        distribution_02 (array-like): Second distribution data

        Returns:
        float: Overlap area metric
        """
        # Compute histograms with normalized probability
        hist_01, edges_01 = np.histogram(distribution_01, bins="auto", density=True)
        hist_02, edges_02 = np.histogram(distribution_02, bins="auto", density=True)

        # Ensure consistent binning across both distributions
        min_edge = min(edges_01[0], edges_02[0])
        max_edge = max(edges_01[-1], edges_02[-1])
        bins = np.linspace(min_edge, max_edge, num=50)

        # Recompute histograms with consistent bins
        hist_01, _ = np.histogram(distribution_01, bins=bins, density=True)
        hist_02, _ = np.histogram(distribution_02, bins=bins, density=True)

        # Compute overlap as minimum of normalized histograms
        overlap = np.sum(np.minimum(hist_01, hist_02)) * (bins[1] - bins[0])

        return overlap

    def plot_and_save(
        self,
        euclid_dist,
        hamming_dist,
        labels,
    ):
        title = "Scatter Plot of Euclidean vs Hamming Distances"

        df = pd.DataFrame(
            np.dstack([euclid_dist, hamming_dist, labels]).squeeze(),
            columns=["euclid_dist", "hamming_dist", "labels"],
        )

        f = sns.jointplot(data=df, x="hamming_dist", y="euclid_dist", hue="labels")

        # Add labels and title
        f.figure.suptitle(title)
        plt.xlabel("Hamming Distance", fontsize=12)
        plt.ylabel("Euclidean Distance", fontsize=12)

        # Show the plot
        plt.tight_layout()
        plt.savefig(f"{self.benchmark_name}.pdf")
        plt.show()

    def report(
        self,
        binarizer: BaseBinarizer,
        n_samples: int,
        euclid_dist,
        hamming_dist,
        labels,
    ):
        bitstring_length = binarizer.info()["bitstring_length"]
        nbits = binarizer.info()["nbits"]

        euclid_auc = self.compute_overlap_auc(
            euclid_dist[labels == True],
            euclid_dist[labels == False],
        )
        hamming_auc = self.compute_overlap_auc(
            hamming_dist[labels == True],
            hamming_dist[labels == False],
        )

        result = f"""
---
Binarizer Benchmarking    
              
+ n_samples {n_samples}
+ bitlength {bitstring_length} bits
+ nbits     {nbits} bits

* Overlapped AUC, Euclidean : {euclid_auc}
* Overlapped AUC, Hamming   : {hamming_auc}
---
"""

        print(result)
        self.plot_and_save(euclid_dist, hamming_dist, labels)
        with open(f"{self.benchmark_name}.txt", "w") as text_file:
            text_file.write(result)


class KeygenReport(BaseBenchmarkReport):
    def __init__(self, benchmark_name: str):
        self.benchmark_name = f"{benchmark_name}"

    def report(self, n_samples: int, false_matchrate: float, false_nonmatchrate: float):
        result = f"""
---
Keygen Benchmarking

+ n_samples {n_samples}

* FMR   : {false_matchrate}
* FNMR  : {false_nonmatchrate}


. FMR should be as low as possible, looking for < 0.001.
. FMNR is expected to be low, but can be compromised, ~ 0.5 would be ok.
---
"""

        print(result)
        with open(f"{self.benchmark_name}.txt", "w") as text_file:
            text_file.write(result)


"""
figures/2025-02-08 10:46:51.745977-binarizer.pdf
figures/2025-02-08 10:46:51.745977-binarizer.txt
figures/2025-02-08 10:46:51.745977-keygen.txt
"""


class BaseBenchmarkCase:
    def __init__(self, binarizer: BaseBinarizer, keygen: BaseKeygen, n_samples: int):
        self.binarizer = binarizer
        self.keygen = keygen
        self.n_samples = n_samples

    def execute(self):
        current_datetime = f"{datetime.datetime.now()}"
        binarizer_report = BinarizerReport(f"results/{current_datetime}-binarizer")
        keygen_report = KeygenReport(f"results/{current_datetime}-keygen")

        pair_loader = PairLoader(self.binarizer)
        all_people = pair_loader.load(self.n_samples)

        print("Done loading.")

        benchmark_binarizer = BenchmarkBinarizer(pair_loader)
        benchmark_keygen = BenchmarkKeygen(self.keygen)

        for _, people_lst in enumerate(all_people):
            benchmark_binarizer.run(people_lst)
            # benchmark_keygen.run(people_lst, pair_loader)

        print("Done running.")

        (euclid_dist, hamming_dist, labels) = benchmark_binarizer.get_result()
        (
            encrypted_total_false_notmatch,
            encrypted_total_false_match,
        ) = benchmark_keygen.get_result()

        binarizer_report.report(
            self.binarizer, self.n_samples, euclid_dist, hamming_dist, labels
        )
        keygen_report.report(
            self.n_samples, encrypted_total_false_match, encrypted_total_false_notmatch
        )

        print("Done reporting.")


class BenchmarkMyFabulousKeygenCase(BaseBenchmarkCase):
    def execute(self):
        current_datetime = f"{datetime.datetime.now()}"
        binarizer_report = BinarizerReport(f"results/{current_datetime}-binarizer")
        keygen_report = KeygenReport(f"results/{current_datetime}-keygen")

        pair_loader = PairLoaderMany(self.binarizer)
        all_people = pair_loader.load(self.n_samples)

        print("Done loading.")

        benchmark_binarizer = BenchmarkFabulousBinarizer(
            pair_loader, self.keygen.registration_index
        )
        benchmark_keygen = BenchmarkMyFabulousKeygen(self.keygen)

        for _, people_lst in enumerate(all_people):
            benchmark_binarizer.run(people_lst)
            benchmark_keygen.run(people_lst, pair_loader)

        print("Done running.")

        (euclid_dist, hamming_dist, labels) = benchmark_binarizer.get_result()
        (
            encrypted_total_false_notmatch,
            encrypted_total_false_match,
        ) = benchmark_keygen.get_result()

        binarizer_report.report(
            self.binarizer, self.n_samples, euclid_dist, hamming_dist, labels
        )
        keygen_report.report(
            self.n_samples, encrypted_total_false_match, encrypted_total_false_notmatch
        )

        print("Done reporting.")


class CaseDROBA(BaseBenchmarkCase):
    def __init__(self, repeat: int = 500):
        binarizer = DROBA(DROBA.Meta().load())
        keygen = DefaultKeygen()
        nsamples = repeat

        super().__init__(
            binarizer,
            keygen,
            nsamples,
        )


class CaseDROBASVC(BaseBenchmarkCase):
    def __init__(self, repeat: int = 500):
        metadata = DROBASVC.Meta()
        metadata.load()
        binarizer = DROBASVC(metadata)
        keygen = DefaultKeygen()
        nsamples = repeat

        super().__init__(
            binarizer,
            keygen,
            nsamples,
        )

class CaseYiming(BaseBenchmarkCase):
    """
    OK Yiming runs but extremely slow. Cannot really use that in practice. 
    The performance is also terrible (I don't know why).
    """

    def __init__(self, repeat: int = 500):
        binarizer = Static()
        keygen = YimingFuzzyExtractor(4096, 500, 128)
        nsamples = repeat

        super().__init__(
            binarizer,
            keygen,
            nsamples,
        )


class CaseMyFuzz(BenchmarkMyFabulousKeygenCase):
    def __init__(self, repeat: int = 500):
        binarizer = Static()
        keygen = MaskThenLockFuzzyExtractor(
            input_length=512,
            key_length=16,
            mask_length=284,
            nonce_len=1,
        )
        nsamples = repeat

        super().__init__(
            binarizer,
            keygen,
            nsamples,
        )

# Not good, only `brgc` is good.
class CaseHigherEntropyBinarizer(BenchmarkMyFabulousKeygenCase):
    def __init__(self, repeat: int = 500):
        binarizer = Static("brgc")
        keygen = MaskThenLockFuzzyExtractor(
            input_length=512,
            key_length=16,
            mask_length=284,
            nonce_len=1,
        )
        nsamples = repeat

        super().__init__(
            binarizer,
            keygen,
            nsamples,
        )


# if __name__ == "__main__":
    # CaseDROBA(500).execute()
    # CaseDROBASVC().execute()
    # CaseYiming(10).execute()
    # CaseMyFuzz(100).execute()
    # CaseHigherEntropyBinarizer(100).execute()


# # Planned to delete soon, temporarily keep here for reference
#
# class BinarizerBenchmark:
#     def __init__(
#         self,
#         binarizer,
#         euclid_threshold: float = 37.5,
#         hamming_threshold: int = 150,
#     ):
#         self.euclid_threshold = euclid_threshold  # nice
#         self.hamming_threshold = hamming_threshold  # nice
#         self.length_bitstring = binarizer.info()["bitstring_length"]
#         self.nbits = binarizer.info()["nbits"]

#         # Initialize binarizer
#         self.binarizer: BaseQuantizer = binarizer

#         self.nsamples = 0

#     def _run_benchmark(self):
#         # For distance analysis
#         euclid_dist = []
#         hamming_dist = []
#         labels = []

#         # For
#         # False Match Rate (FMR) = ACCEPTED false login / samples (for false logins)
#         # False Non-Match Rate (FNMR) = REJECTED true login / samples  (for true logins)
#         L_bytes = int(self.length_bitstring / 8)
#         extractor = DefaultKeygen()

#         euclid_total_false_nonmatch = 0  # For true logins
#         hamming_total_false_nonmatch = 0
#         encrypted_total_false_notmatch = 0

#         euclid_total_false_match = 0  # For false logins
#         hamming_total_false_match = 0
#         encrypted_total_false_match = 0

#         gen_time = 0
#         rep_time = 0

#         for i, people_lst in enumerate(self.all_people):
#             start = time.time()

#             # Get a true identity and a false identity
#             true_identity = 0
#             false_identity = 1

#             # Get 02 random vectors from true identity (for signup and login)
#             embeddings = people_lst[true_identity][-1]["embeddings"]
#             np_boolarrs = [
#                 self.binarizer.binarise_asbool(embedding) for embedding in embeddings
#             ]
#             length = people_lst[true_identity][-1]["n"]

#             select_signup = randint(0, length - 1)
#             select_login = randint(0, length - 1)
#             while select_login == select_signup:
#                 select_login = randint(0, length - 1)

#             signup_np_boolarr = np_boolarrs[select_signup]
#             login_np_boolarr = np_boolarrs[select_login]

#             signup_np_emb = embeddings[select_signup]
#             login_np_emb = embeddings[select_login]

#             signup_raw = to_bytestring(signup_np_boolarr, L_bytes)
#             login_raw = to_bytestring(login_np_boolarr, L_bytes)

#             gen_start = time.time()
#             signup_key, helper = extractor._generate(signup_raw)
#             gen_end = time.time()
#             gen_time += gen_end - gen_start

#             # distance
#             individual_euclid_dist = np.linalg.norm(signup_np_emb - login_np_emb)
#             individual_hamming_dist = np.count_nonzero(
#                 signup_np_boolarr != login_np_boolarr
#             )
#             rep_start = time.time()
#             rep = extractor._reproduce(login_raw, helper)
#             rep_end = time.time()

#             rep_time += rep_end - rep_start
#             # Record true login results
#             encrypted_total_false_notmatch += int(not (rep == signup_key))
#             euclid_total_false_nonmatch += int(
#                 individual_euclid_dist >= self.euclid_threshold
#             )
#             hamming_total_false_nonmatch += int(
#                 individual_hamming_dist >= self.hamming_threshold
#             )

#             # Record result
#             euclid_dist += [individual_euclid_dist]
#             hamming_dist += [individual_hamming_dist]
#             labels += [True]

#             # Get 01 random vectors from false identity (for false login)
#             embeddings = people_lst[false_identity][-1]["embeddings"]
#             np_boolarrs = [
#                 self.binarizer.binarise_asbool(embedding) for embedding in embeddings
#             ]
#             length = people_lst[false_identity][-1]["n"]

#             select_login = randint(0, length - 1)
#             login_np_boolarr = np_boolarrs[select_login]
#             login_np_emb = embeddings[select_login]
#             login_raw = to_bytestring(login_np_boolarr, L_bytes)

#             # distance
#             individual_euclid_dist = np.linalg.norm(signup_np_emb - login_np_emb)
#             individual_hamming_dist = np.count_nonzero(
#                 signup_np_boolarr != login_np_boolarr
#             )
#             # Record false login results
#             rep = extractor._reproduce(login_raw, helper)
#             encrypted_total_false_match += int(rep == signup_key)
#             euclid_total_false_match += int(
#                 individual_euclid_dist < self.euclid_threshold
#             )
#             hamming_total_false_match += int(
#                 individual_hamming_dist < self.hamming_threshold
#             )

#             # Record result, note that it is still the same signup_emb and boolarr.
#             euclid_dist += [individual_euclid_dist]
#             hamming_dist += [individual_hamming_dist]
#             labels += [False]

#             end = time.time()
#             print(f"Finished one sample ({i}\{len(self.all_people)})...{end - start} s")

#         euclid_total_false_nonmatch /= float(self.nsamples)  # For true logins ()
#         hamming_total_false_nonmatch /= float(self.nsamples)
#         encrypted_total_false_notmatch /= float(self.nsamples)
#         nonmatch = (
#             euclid_total_false_nonmatch,
#             hamming_total_false_nonmatch,
#             encrypted_total_false_notmatch,
#         )

#         euclid_total_false_match /= float(self.nsamples)  # For false logins ()
#         hamming_total_false_match /= float(self.nsamples)
#         encrypted_total_false_match /= float(self.nsamples)
#         match = (
#             euclid_total_false_match,
#             hamming_total_false_match,
#             encrypted_total_false_match,
#         )

#         gen_time /= float(self.nsamples)
#         rep_time /= float(self.nsamples)

#         # For reports
#         self.gen_time = gen_time
#         self.rep_time = rep_time

#         self.nonmatch = nonmatch
#         self.match = match

#         self.euclid_dist = np.array(euclid_dist)
#         self.hamming_dist = np.array(hamming_dist)
#         self.labels = np.array(labels)

#     def _report_benchmark(self):
#         euclid_overlapped = compute_overlap_auc(
#             self.euclid_dist[self.labels == True],
#             self.euclid_dist[self.labels == False],
#         )
#         hamming_overlapped = compute_overlap_auc(
#             self.hamming_dist[self.labels == True],
#             self.hamming_dist[self.labels == False],
#         )

#         title = f"""
# L={self.length_bitstring} bits, n_bits={self.nbits}, n_samples={self.nsamples}
# euclid_overlapped={"{:0.4f}".format(euclid_overlapped)},
# hamming_overlapped={"{:0.4f}".format(hamming_overlapped)}
# """

#         nonmatch_str = "\t".join([str(x) for x in self.nonmatch])
#         match_str = "\t".join([str(x) for x in self.match])

#         print(
#             f"""
#     Benchmark Binarizer {self.binarizer}-{self.length_bitstring} bits-{self.nbits} nbits-{self.nsamples} nsamples

#         Time
#         ---
#         Fuzzy Extractor (Gen):  {self.gen_time} s
#         Fuzzy Extractor (Rep):  {self.rep_time} s

#         Storage
#         ---
#         Binarizer:                      {os.path.getsize("./binarizer.pickle")} KB
#         Helper Object (per enrollment): Not defined.

#         AUC
#         ---
#         overlapped_euclid: {euclid_overlapped}
#         overlapped_hamming: {hamming_overlapped}

#         Match Rate
#         ---
#                 euclid  hamming encrypted
#         FNR     {nonmatch_str}
#         FMR     {match_str}
#         FNR should be low, ok to be high.
#         FMR (false acceptance rate) should be low at all cost.

#         Figure
#         ---
#         saved.
#             """
#         )

#         print("Showing figure...")
#         create_figure(self.euclid_dist, self.hamming_dist, self.labels, title=title)

#     def run(self):
#         print(
#             """
# Started testing...
# """
#         )
#         self._loadtar()
#         self._load_benchmark()
#         self._run_benchmark()
#         self._report_benchmark()
