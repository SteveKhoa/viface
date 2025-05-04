import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from benchmark.constant import N_REPEAT, TAR_EMBEDDINGS_FILE_PATH, DROBA_PATH
from benchmark.utils_benchmark_case import BenchmarkCase
from benchmark.utils_data_transform_embeddings_by_pair import DataTransformEmbeddingsByPair
from benchmark.utils_data_extract_tar import DataExtractTarToFileDictionary
from lib.biocryp.binarizers import DROBA
from lib.biocryp.keygen import DefaultKeygen


class CaseDROBA(BenchmarkCase):
    def __init__(self, repeat: int = 500):
        binarizer = DROBA(DROBA.Meta(path=DROBA_PATH).load())
        keygen = DefaultKeygen()
        data_extract = DataExtractTarToFileDictionary(TAR_EMBEDDINGS_FILE_PATH)

        file_dictionary = data_extract.extract()

        super().__init__(
            binarizer,
            keygen,
            turn_on_keygen_flag=False,
            n_samples=repeat,
            benchmark_transform=DataTransformEmbeddingsByPair(repeat, file_dictionary),
        )


if __name__ == "__main__":
    """
    This benchmark does not intend for keygen benchmarking. So use any keygen does not matter
    """
    CaseDROBA(N_REPEAT).execute()
