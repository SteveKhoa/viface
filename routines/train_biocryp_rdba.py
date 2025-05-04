import numpy as np
import os
from sklearn import tree, decomposition
import time
import re
from random import randint
import random
import sklearn
import numpy as np
import os
from collections import defaultdict
import heapq
import re
import random
from lib.biocryp.quantizers import AnyQuantizer
from lib.biocryp.binarizers import DROBA, DROBASVC
import sklearn.preprocessing
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


random.seed(0)


class Dataset:
    def __init__(self, src):
        self.src = src
        self.all_people = {}
        self._get_meta()

    def _get_meta(self):
        """
        Get metadata of the files (filenames)
        """
        if bool(self.all_people) is False:
            filenames = list(filter(lambda x: x != ".DS_Store", os.listdir(self.src)))
            self.n_samples = len(filenames)
            self.filenames = filenames

            all_people = defaultdict(int)
            for facename in filenames:
                person_name = tuple(facename.split("_")[:-1])
                rejoined_name = "_".join(person_name)
                all_people[rejoined_name] += 1

            self.all_people = all_people
            self.filtered = all_people
            self.n_samples_per_identity = min(list(all_people.values()))
        else:
            pass

    def filter_most_common(self, k_top):
        """
        One of the filtering functions
        """

        top = heapq.nlargest(k_top, self.filtered.items(), key=lambda x: x[-1])
        top = dict(top)

        n_samples_per_identity = min(list(top.values()))
        n_samples = n_samples_per_identity * len(top.keys())

        self.filtered = top
        self.n_samples = n_samples
        self.n_samples_per_identity = n_samples_per_identity
        return self

    def filter_complement(self):
        key_table = {}
        for x in self.filtered.items():
            key_table[x[0]] = 1

        self.filtered = dict(
            filter(lambda x: not (x[0] in key_table), self.all_people.items())
        )
        return self

    def get_meta(self):
        return [{"name": x[0], "n_samples": x[1]} for x in self.filtered.items()]

    def _get_samples(self):
        """
        Returns paths to filtered files.
        """
        people_names = self.filtered.keys()
        n_samples_per_identity = self.n_samples_per_identity

        people_filenames = defaultdict(list)
        for person_name in people_names:
            files = list(
                filter(lambda x: re.match(f"{person_name}.*", x), self.filenames)
            )
            first_n_samples = files[:n_samples_per_identity]
            for filename in first_n_samples:
                people_filenames[person_name] += [os.path.join(self.src, filename)]

        self.samples = people_filenames

    def filter_random(self):
        rand_idx = random.randint(0, len(self.filtered))
        x = list(self.filtered.items())[rand_idx]
        self.filtered = dict([x])

    def load(self):
        self._get_samples()

        datasets = []
        for name, filenames in self.samples.items():
            identity = {
                "name": name,
                "samples": [np.load(filename) for filename in filenames],
                "n_samples": len(filenames),
            }
            datasets += [identity]
        return datasets, self.n_samples


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        L=128,
        pca_ncomponents=32,
        dectree_maxdepth=14,
        nbits=16,
        feature_as_pca=False,
    ):
        self.dataset = dataset
        self.bitstring_length = L
        self.pca_ncomps = pca_ncomponents
        self.dectree_maxdepth = dectree_maxdepth
        self.nbits = nbits
        self.feature_aspca = feature_as_pca

    def run(self):
        print(
            f"""
Training started...

Bitlength:          {self.bitstring_length} bits
Nbits:              {self.nbits} bits
PCA components:     {self.pca_ncomps}
Dectree depth:      {self.dectree_maxdepth}
Feature as PCA:     {self.feature_aspca}
"""
        )
        self.dataset, self.nsamples = self.dataset.load()
        self._build_heuristics()._analyze_stats()._compute_reliability()._allocate_bits()

        DROBA.Meta().save(
            self.pca,
            self.strategy_classifier,
            self.max_pdfs,
            self.min_pdfs,
            self.multi_nd_mask,
            self.feature_aspca,
        )
        print(
            """
Training done...
"""
        )

    def _build_heuristics(self):
        print("Building heuristics (PCA and DECTREE)...", end="")
        start = time.time()

        all_samples = []
        labels = []
        for idx, identity in enumerate(self.dataset):
            samples = identity["samples"]
            all_samples += samples
            labels += [idx] * (len(samples))

        # Fitting Decision Tree
        X = all_samples
        X = sklearn.preprocessing.normalize(X)

        # Reduce dimensions for better classfication
        pca = decomposition.PCA(n_components=self.pca_ncomps, svd_solver="randomized")
        pca = pca.fit(X)

        X = pca.transform(X)
        Y = labels
        clf = tree.DecisionTreeClassifier(max_depth=self.dectree_maxdepth)
        best_extracted_X = X
        clf = clf.fit(best_extracted_X, Y)

        # Use PCA to chop down unecessary dimensions (with low variance)
        new_datasets = []
        for idx, identity in enumerate(self.dataset):
            embeddings = identity["samples"]
            embeddings = sklearn.preprocessing.normalize(embeddings)
            if self.feature_aspca:
                new_samples = list(pca.transform(embeddings))
            else:
                new_samples = list(embeddings)

            new_datasets += [{"name": identity["name"], "samples": new_samples}]

        end = time.time()
        print(f"{end - start} s")

        self.dataset = new_datasets
        self.strategy_classifier = clf
        self.pca = pca

        return self

    def _analyze_stats(self):
        print("Analyzing dataset statistics...", end="")
        start = time.time()

        individual_means = []
        individual_within_variances = []
        for idx, identity in enumerate(self.dataset):
            samples = identity["samples"]

            individual_means += [np.mean(samples, axis=0)]
            individual_within_variances += [np.var(np.array(samples).T, axis=1)]
        individual_means = np.array(
            individual_means
        )  # each array index is directly related to an identity, don't change it
        individual_within_variances = np.array(individual_within_variances)

        # grand_mean = np.mean(individual_means, axis=0)  # deprecated
        between_variance = np.var(individual_means.T, axis=1)
        signoise_ratios = np.array(between_variance / individual_within_variances)
        # avg_signoise = np.mean(signoise_ratios, axis=0)  # lets use average as an aggregation formula
        nd_ranks = (-signoise_ratios).argsort()

        end = time.time()
        print(f"{end - start} s")

        self.nd_ranks = nd_ranks
        return self

    def _compute_reliability(self):
        """
        This implementation uses equal-width quantization instead of equal-probable in the original paper.

        This paper[https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/2577914/DeepFace_Binarisation.pdf?sequence=2&isAllowed=y]
        believes equal-width has better performance. And equal-width compromises security instead of privacy (binarization runs on local
        machine, makes perfect sense to trade off security to ensure privacy. <-- yes RDBA2012 talked about this too)

        `all_samples` are all concatenated samples from all identities
        """
        print("Computing reliability...", end="")
        start = time.time()

        all_samples = []
        for identity in self.dataset:
            samples = identity["samples"]
            all_samples += samples
        all_samples = np.array(all_samples)  # to do numpy ops
        self.min_pdfs = np.min(all_samples, axis=0)
        self.max_pdfs = np.max(all_samples, axis=0)
        n_samples = all_samples.shape[0]

        # Initial quantization using equal-width, Gray Code.
        nd_reliability_weights = []
        # nd_bitstrings = []
        q = AnyQuantizer(nbits=self.nbits)
        for dim in range(all_samples.shape[1]):
            # Quantization
            labels = np.array(
                list(
                    map(
                        lambda val: q.quantize(
                            val, self.min_pdfs[dim], self.max_pdfs[dim]
                        ),
                        all_samples[:, dim],
                    )
                )
            )

            # Count 1s in each bit position in Gray Code labels
            # This is heavily dependent on gray code's length
            bitstrings = np.array(
                [
                    list(
                        map(
                            lambda x: int(x),
                            list(
                                "{0:0{1}b}".format(label, self.nbits)
                            ),  # cast it to "binary" string
                        )
                    )
                    for label in labels
                ]
            )

            dimensional_weights = np.sum(bitstrings, axis=0) / float(n_samples)
            complement_weights = 1.0 - dimensional_weights

            # Rescale to [0.5,1]
            rescaled = np.max(
                np.array([dimensional_weights, complement_weights]),
                axis=0,
            )

            nd_reliability_weights += [list(rescaled)]
            # nd_bitstrings += [bitstrings]

        end = time.time()
        print(f"{end - start} s")

        self.nd_reliability_weights = nd_reliability_weights
        return self

    def _allocate_bits(self):
        print(f"Allocating masks...")
        total_start = time.time()

        def __hehe(ranks):
            result_nd_mask = None
            n_max = len(self.nd_reliability_weights[0])
            n_dims = len(ranks)

            for x in range(0, int(np.ceil(self.nsamples / 2)) + 1):
                reliability_threshold = 1 - float(x) / float(self.nsamples)

                nd_mask = [[False] * n_max] * n_dims
                n_bits = 0
                for idx in ranks:
                    dimensional_weights = self.nd_reliability_weights[idx]
                    mask = []
                    for w_id in range(n_max):
                        if dimensional_weights[w_id] > reliability_threshold:
                            mask += [True]
                        else:
                            mask += [False] * (n_max - w_id)
                            break
                    n_maskbits = np.sum(mask)
                    n_bits += n_maskbits

                    if n_bits < self.bitstring_length:
                        nd_mask[idx] = mask
                    else:
                        n_bits_to_allocate = self.bitstring_length - (
                            n_bits - n_maskbits
                        )
                        mask = ([True] * n_bits_to_allocate) + (
                            [False] * (n_max - n_bits_to_allocate)
                        )
                        nd_mask[idx] = mask
                        result_nd_mask = np.array(nd_mask)
                        return result_nd_mask
            return result_nd_mask

        multi_nd_mask = []
        for id, ranks in enumerate(self.nd_ranks):
            start = time.time()

            result_nd_mask = __hehe(ranks)

            end = time.time()
            print(f"\tBitmask created ({id}\{len(self.nd_ranks)})...{end - start} s")
            multi_nd_mask += [result_nd_mask]

        total_end = time.time()
        print(f"Allocating masks...{total_end - total_start} s")
        self.multi_nd_mask = multi_nd_mask


class TrainerSVC(Trainer):
    def __init__(
        self,
        dataset: Dataset,
        kernel,
        L=128,
        nbits=16,
    ):
        self.dataset = dataset
        self.bitstring_length = L
        self.kernel = kernel
        self.nbits = nbits

    def run(self):
        print(
            f"""
Training started...

Bitlength:          {self.bitstring_length} bits
Nbits:              {self.nbits} bits
SVC Kernel:         {self.kernel}
"""
        )
        self.dataset, self.nsamples = self.dataset.load()
        self._build_heuristics()._analyze_stats()._compute_reliability()._allocate_bits()

        DROBASVC.Meta().save(
            self.svc,
            self.max_pdfs,
            self.min_pdfs,
            self.multi_nd_mask,
        )
        print(
            """
Training done...
"""
        )

    def _build_heuristics(self):
        print("Building heuristics (SVC)...", end="")
        start = time.time()

        # Hehe
        all_samples = []
        labels = []
        for idx, identity in enumerate(self.dataset):
            samples = identity["samples"]
            all_samples += samples
            labels += [idx] * (len(samples))

        svm_model = SVC(kernel=self.kernel)
        X = all_samples
        X = sklearn.preprocessing.normalize(X)
        Y = labels
        svm_model.fit(X, Y)

        new_datasets = []
        for idx, identity in enumerate(self.dataset):
            embeddings = identity["samples"]
            embeddings = sklearn.preprocessing.normalize(embeddings)
            new_samples = list(embeddings)
            new_datasets += [{"name": identity["name"], "samples": new_samples}]

        end = time.time()
        print(f"{end - start} s")

        self.dataset = new_datasets
        self.svc = svm_model

        return self


def train():
    ds = Dataset("./lfw.embeddings")
    ds.filter_most_common(20)

    tr = Trainer(ds, 4096, 32, 14, 32, False)
    tr.run()


# def option1():
#     most_common = 20
#     pca_ncomps = 64
#     dectree_maxdepth = 89
#     nsamples = 500

#     """Results
#         bitlength   16      32      64
#     maxdepth

#     5               0.519   0.519   0.519
#     55              0.67    0.70    0.68
#     89              0.68    0.67    0.68
#     144             0.69    0.67    0.62

#     (see figure/option1.txt)
#     """

#     ds = Dataset("./lfw.embeddings")
#     ds.filter_most_common(most_common)
#     dataset, _ = ds.load()

#     # Hehe
#     all_samples = []
#     labels = []
#     for idx, identity in enumerate(dataset):
#         samples = identity["samples"]
#         all_samples += samples
#         labels += [idx] * (len(samples))

#     X = all_samples
#     X = sklearn.preprocessing.normalize(X)

#     clf = tree.DecisionTreeClassifier(max_depth=dectree_maxdepth)
#     # Reduce dimensions for better classfication
#     pca = decomposition.PCA(n_components=pca_ncomps)
#     pca = pca.fit(X)

#     X = pca.transform(X)
#     Y = labels
#     clf = clf.fit(X, Y)

#     print("Train PCA and Dectree done...")
#     print()

#     #
#     #
#     #
#     #
#     #

#     from biocryp_benchmark import BinarizerBenchmark

#     b = BinarizerBenchmark(nsamples=nsamples)
#     b._load_benchmark()

#     true_positives = 0
#     false_positives = 0

#     for i, people_lst in enumerate(b.all_people):
#         # Get a true identity and a false identity
#         true_identity = 0
#         false_identity = 1

#         # Get 02 random vectors from true identity (for signup and login)
#         embeddings = people_lst[true_identity][-1]["embeddings"]
#         length = people_lst[true_identity][-1]["n"]

#         select_signup = randint(0, length - 1)
#         select_login = randint(0, length - 1)
#         while select_login == select_signup:
#             select_login = randint(0, length - 1)

#         signup_np_emb = sklearn.preprocessing.normalize(
#             np.expand_dims(embeddings[select_signup], axis=0)
#         )
#         signup_pca = pca.transform(signup_np_emb)
#         signup_np_emb = np.squeeze(signup_np_emb)
#         login_np_emb = sklearn.preprocessing.normalize(
#             np.expand_dims(embeddings[select_login], axis=0)
#         )
#         login_pca = pca.transform(login_np_emb)
#         login_np_emb = np.squeeze(login_np_emb)

#         # Run classification
#         signup_category = clf.predict(signup_pca)[0]
#         login_category = clf.predict(login_pca)[0]

#         # Compute some metric
#         true_positives += int(signup_category == login_category)

#         # print("Yes", signup_category, login_category)

#         # Get 01 random vectors from false identity (for false login)
#         embeddings = people_lst[false_identity][-1]["embeddings"]
#         length = people_lst[false_identity][-1]["n"]
#         select_login = randint(0, length - 1)
#         login_np_emb = sklearn.preprocessing.normalize(
#             np.expand_dims(embeddings[select_login], axis=0)
#         )
#         login_pca = pca.transform(login_np_emb)
#         login_np_emb = np.squeeze(login_np_emb)

#         # Run classfication
#         login_category = clf.predict(login_pca)[0]
#         false_positives += int(signup_category == login_category)

#         # print("No", signup_category, login_category)

#     avg_true_positives = true_positives / float(len(b.all_people))
#     avg_false_positives = false_positives / float(len(b.all_people))
#     precision = avg_true_positives / (avg_true_positives + avg_false_positives)
#     print()
#     print(f"pca_ncomps={pca_ncomps}, dectree_maxdepth={dectree_maxdepth}")
#     print("True positives:", avg_true_positives)
#     print("False positives:", avg_false_positives)
#     print("Precision:", precision)


# def option2():
#     most_common = 1000
#     kernel = "sigmoid"

#     """
#     Sigmoid SVC is obviously better than PCA+DECTREE.

#     SVM, kernel sigmoid
#     True positives  (True)   : 0.306
#     False positives (Error)  : 0.094
#     Precision: 0.7649999999999999

#     while, the best of PCA+DECTREE achieves only 0.7 max

#     ==> that's why we will choose sigmoid SVC

#     HOWEVER, it is TERRIBLE when integrated with Binarizer.
#     In fact, Binarizer works better if we just don't classify anything at all (option 3, static binarizer).
#     By being terrible, I mean: Hamming overlapped is unacceptably high.
#     """

#     ds = Dataset("./lfw.embeddings")
#     ds.filter_most_common(most_common)

#     dataset, _ = ds.load()

#     # Hehe
#     all_samples = []
#     labels = []
#     for idx, identity in enumerate(dataset):
#         samples = identity["samples"]
#         all_samples += samples
#         labels += [idx] * (len(samples))

#     svm_model = SVC(kernel=kernel, tol=2.5)
#     X = all_samples
#     X = sklearn.preprocessing.normalize(X)
#     Y = labels
#     svm_model.fit(X, Y)

#     print("Train PCA and Dectree done...")
#     print()

#     #
#     #
#     #
#     #
#     #

#     from biocryp_benchmark import BinarizerBenchmark

#     metadata = DROBASVC.Meta()
#     metadata.load()
#     binarizer = DROBASVC(metadata)
#     b = BinarizerBenchmark(binarizer=binarizer, nsamples=500)
#     b._load_benchmark()

#     true_positives = 0
#     false_positives = 0

#     for i, people_lst in enumerate(b.all_people):
#         # Get a true identity and a false identity
#         true_identity = 0
#         false_identity = 1

#         # Get 02 random vectors from true identity (for signup and login)
#         embeddings = people_lst[true_identity][-1]["embeddings"]
#         length = people_lst[true_identity][-1]["n"]

#         select_signup = randint(0, length - 1)
#         select_login = randint(0, length - 1)
#         while select_login == select_signup:
#             select_login = randint(0, length - 1)

#         signup_np_emb = np.expand_dims(embeddings[select_signup], axis=0)
#         login_np_emb = np.expand_dims(embeddings[select_login], axis=0)
#         signup_np_emb = sklearn.preprocessing.normalize(signup_np_emb)
#         login_np_emb = sklearn.preprocessing.normalize(login_np_emb)

#         # Run classification
#         signup_category = svm_model.predict(signup_np_emb)[0]
#         login_category = svm_model.predict(login_np_emb)[0]

#         # Compute some metric
#         true_positives += int(signup_category == login_category)

#         # Get 01 random vectors from false identity (for false login)
#         embeddings = people_lst[false_identity][-1]["embeddings"]
#         length = people_lst[false_identity][-1]["n"]
#         select_login = randint(0, length - 1)
#         login_np_emb = np.expand_dims(embeddings[select_login], axis=0)
#         login_np_emb = sklearn.preprocessing.normalize(login_np_emb)

#         # Run classfication
#         login_category = svm_model.predict(login_np_emb)[0]
#         false_positives += int(signup_category == login_category)

#     avg_true_positives = true_positives / float(len(b.all_people))
#     avg_false_positives = false_positives / float(len(b.all_people))
#     precision = avg_true_positives / (avg_true_positives + avg_false_positives)
#     print()
#     print(f"SVM, kernel {kernel}")
#     print("True positives  (True)   :", avg_true_positives)
#     print("False positives (Error)  :", avg_false_positives)
#     print("Precision:", precision)


if __name__ == "__main__":
    train()
    # option1()
