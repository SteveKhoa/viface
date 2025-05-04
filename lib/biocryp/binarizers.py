from sklearn.svm import SVC
import numpy as np
import sklearn.preprocessing
import sklearn
import pickle
from lib.biocryp import vars, quantizers
from typing import List
from typing import Optional
import abc


class BaseBinarizer(object):
    def __init__(self, bitstring_length: int, nbits: int):
        self.bitstring_length = bitstring_length
        self.nbits = nbits

    def _to_bits(self, binary_label: int) -> List[bool]:
        return list(
            map(lambda x: x == "1", list("{0:0{1}b}".format(binary_label, self.nbits)))
        )

    def _to_bytestring(self, bool_arr):
        x = sum(map(lambda x: x[1] << x[0], enumerate(bool_arr)))
        bin_ = int(x).to_bytes(length=vars.BYTE_LENGTH, byteorder="big", signed=True)
        return bin_

    def info(self):
        """General information about the quantizer"""
        return {"bitstring_length": self.bitstring_length, "nbits": self.nbits}

    @abc.abstractmethod
    def binarise_asbool(self, embedding) -> List[bool]:
        """Return binarised biometric as array of booleans. This is handy for downstream tasks."""
        pass

    @abc.abstractmethod
    def binarise(self, embedding) -> bytes:
        """Return binarised biometric as bytes stream. For easier interpretation, use `binarise_asbool()`."""
        pass


class DROBASVC(BaseBinarizer):
    """
    A variant of DROBA that uses SVC Sigmoid instead of PCA+DECTREE.
    Experiments show that SVC Sigmoid is much better than PCA+DECTREE,
    while does not need to tune the hyperparameters too much.
    """

    class Meta:
        def __init__(self):
            pass

        def save(self, svc: SVC, max_pdfs, min_pdfs, nd_reliability_mask):
            datastructure = {
                "svc": svc,
                "max_pdfs": max_pdfs,
                "min_pdfs": min_pdfs,
                "nd_reliability_mask": nd_reliability_mask,
            }
            with open(vars.BINARIZER_PATH, "wb") as write_f:
                pickle.dump(datastructure, write_f, pickle.HIGHEST_PROTOCOL)

        def load(self):
            with open(vars.BINARIZER_PATH, "rb") as read_f:
                datastructure = pickle.load(read_f)
            self.svc: SVC = datastructure["svc"]
            self.max_pdfs = datastructure["max_pdfs"]
            self.min_pdfs = datastructure["min_pdfs"]
            self.nd_reliability_mask = datastructure["nd_reliability_mask"]

    def __init__(self, metadata: Meta):
        self.svc = metadata.svc
        self.max_pdfs = metadata.max_pdfs
        self.min_pdfs = metadata.min_pdfs
        self.nd_reliability_mask = metadata.nd_reliability_mask

        self.bitstring_length = np.count_nonzero(metadata.nd_reliability_mask[0])
        self.nbits = self.nd_reliability_mask[0].shape[1]
        super().__init__(self.bitstring_length, self.nbits)

        self.quantizer = quantizers.AnyQuantizer(coding_scheme="lssc", nbits=self.nbits)

    def _core_binarise(self, embedding) -> np.ndarray:
        embedding = np.expand_dims(embedding, axis=0)
        embedding = sklearn.preprocessing.normalize(embedding)

        # Choose the "best" reliability mask for embedding coding.
        category = self.svc.predict(embedding)[0]
        reliability_mask = self.nd_reliability_mask[category]

        embedding = np.squeeze(embedding)

        # Quantization
        binarised = []
        for dim in range(embedding.shape[0]):
            binary_label = self.quantizer.quantize(
                embedding[dim],
                self.min_pdfs[dim],
                self.max_pdfs[dim],
            )
            bitstring = self._to_bits(binary_label)
            binarised += [bitstring]

        # Concat and mask the bitstring
        concatenated = np.array(binarised)[np.array(reliability_mask)]
        return concatenated

    def binarise_asbool(self, embedding) -> List[bool]:
        return self._core_binarise(embedding)

    def binarise(self, embedding) -> bytes:
        boolarr = self._core_binarise(embedding)
        bytestring = self._to_bytestring(boolarr)
        return bytestring


class DROBA(BaseBinarizer):
    class Meta:
        def __init__(self, path: Optional[str] = None):
            if path is not None:
                self.path = path
            else:
                self.path = vars.BINARIZER_PATH

        def save(
            self,
            pca,
            dectree,
            max_pdfs,
            min_pdfs,
            nd_reliability_mask,
            feature_as_pca=False,
        ):
            datastructure = {
                "pca": pca,
                "dectree": dectree,
                "max_pdfs": max_pdfs,
                "min_pdfs": min_pdfs,
                "nd_reliability_mask": nd_reliability_mask,
                "feature_as_pca": feature_as_pca,
            }
            with open(self.path, "wb") as write_f:
                pickle.dump(datastructure, write_f, pickle.HIGHEST_PROTOCOL)

            return self

        def load(self):
            with open(self.path, "rb") as read_f:
                datastructure = pickle.load(read_f)
            self.pca = datastructure["pca"]
            self.dectree = datastructure["dectree"]
            self.max_pdfs = datastructure["max_pdfs"]
            self.min_pdfs = datastructure["min_pdfs"]
            self.nd_reliability_mask = datastructure["nd_reliability_mask"]
            self.feature_as_pca = datastructure["feature_as_pca"]

            return self

    def __init__(self, metadata: Meta):
        self.pca = metadata.pca
        self.dectree = metadata.dectree
        self.max_pdfs = metadata.max_pdfs
        self.min_pdfs = metadata.min_pdfs
        self.nd_reliability_mask = metadata.nd_reliability_mask
        self.bitstring_length = np.count_nonzero(metadata.nd_reliability_mask[0])
        self.feature_as_pca = metadata.feature_as_pca

        self.nbits = self.nd_reliability_mask[0].shape[1]
        self.quantizer = quantizers.AnyQuantizer(nbits=self.nbits)

        super().__init__(self.bitstring_length, self.nbits)

    def _core_binarise(self, embedding) -> np.ndarray:
        embedding = np.expand_dims(embedding, axis=0)
        embedding = sklearn.preprocessing.normalize(embedding)  # ??????
        pca_vector = self.pca.transform(embedding)

        if self.feature_as_pca:
            embedding = pca_vector
        embedding = np.squeeze(embedding)

        # Choose the "best" reliability mask for embedding coding.
        best_extracted_pca_vector = pca_vector
        category = self.dectree.predict(best_extracted_pca_vector)[0]
        reliability_mask = self.nd_reliability_mask[0]

        # Quantization
        binarised = []
        for dim in range(embedding.shape[0]):
            binary_label = self.quantizer.quantize(
                embedding[dim],
                self.min_pdfs[dim],
                self.max_pdfs[dim],
            )
            bitstring = self._to_bits(binary_label)
            binarised += [bitstring]

        # Concat and mask the bitstring
        concatenated = np.array(binarised)[np.array(reliability_mask)]
        return concatenated

    def binarise_asbool(self, embedding) -> List[bool]:
        return self._core_binarise(embedding)

    def binarise(self, embedding) -> bytes:
        boolarr = self._core_binarise(embedding)
        bytestring = self._to_bytestring(boolarr)
        return bytestring


class Static(BaseBinarizer):
    """
    Static binarizer.
    """

    def __init__(self, coding_scheme = "lssc"):
        bitstring_length = vars.BYTE_LENGTH * 8
        ndims = vars.NDIMS
        self.nbits = int(bitstring_length / ndims)
        self.bitstring_length = bitstring_length
        self.quantizer = quantizers.AnyQuantizer(coding_scheme, nbits=self.nbits)

    def _core_binarise(self, embedding):
        assert self.nbits * embedding.shape[0] == self.bitstring_length
        # Quantization
        binarised = []
        for dim in range(embedding.shape[0]):
            binary_label = self.quantizer.quantize(
                embedding[dim],
                -0.2,
                0.2,
            )
            bitstring = self._to_bits(binary_label)
            binarised += bitstring

        concatenated = np.array(binarised)
        return concatenated

    def binarise_asbool(self, embedding) -> List[bool]:
        return self._core_binarise(embedding)

    def binarise(self, embedding) -> bytes:
        boolarr = self._core_binarise(embedding)
        bytestring = self._to_bytestring(boolarr)
        return bytestring


if __name__ == "__main__":
    q = quantizers.AnyQuantizer("plssc", 2)
    x = q.quantize(44442.2, 0.1, 199.4)
    print(q.coding_table)
    print(x)
