import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
from lib.biocryp.binarizers import BaseBinarizer
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import abc
from benchmark import utils_benchmark_report_templates


class BaseBenchmarkReport(object):
    @abc.abstractmethod
    def report(self):
        pass


class BinarizerReport(BaseBenchmarkReport):
    def __init__(self, path_to_save: str):
        self.path_to_save = path_to_save

    def _compute_overlap_auc(self, distribution_01, distribution_02):
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

    def _plot_and_save(
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
        plt.savefig(os.path.join(f"{self.path_to_save}", "binarizer.pdf"))
        plt.show()

    def report(
        self,
        binarizer: BaseBinarizer,
        n_samples: int,
        euclid_dist,
        hamming_dist,
        labels,
        execution_time,
    ):
        bitstring_length = binarizer.info()["bitstring_length"]
        nbits = binarizer.info()["nbits"]

        euclid_auc = self._compute_overlap_auc(
            euclid_dist[labels == True],
            euclid_dist[labels == False],
        )
        hamming_auc = self._compute_overlap_auc(
            hamming_dist[labels == True],
            hamming_dist[labels == False],
        )

        result = utils_benchmark_report_templates.BINARIZER_REPORT_TEMPLATE % {
            "n_samples": n_samples,
            "bitstring_length": bitstring_length,
            "nbits": nbits,
            "euclid_auc": euclid_auc,
            "hamming_auc": hamming_auc,
            "execution_time": execution_time,
        }

        print(result)
        self._plot_and_save(euclid_dist, hamming_dist, labels)
        with open(os.path.join(f"{self.path_to_save}", "binarizer.txt"), "w") as text_file:
            text_file.write(result)


class KeygenReport(BaseBenchmarkReport):
    def __init__(self, path_to_save: str):
        self.path_to_save = path_to_save

    def report(self, n_samples: int, false_matchrate: float, false_nonmatchrate: float, total_registration_time: float, total_login_time: float):
        result = utils_benchmark_report_templates.KEYGEN_REPORT_TEMPLATE % {
            "n_samples": n_samples,
            "false_matchrate": false_matchrate,
            "false_nonmatchrate": false_nonmatchrate,
            "total_registration_time": "%1.4f" % round(total_registration_time, 4),
            "total_login_time": "%1.4f" % round(total_login_time, 4),
        }

        print(result)
        with open(os.path.join(f"{self.path_to_save}", "keygen.txt"), "w") as text_file:
            text_file.write(result)
