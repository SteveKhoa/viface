import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from benchmark.constant import N_REPEAT, TAR_EMBEDDINGS_FILE_PATH, BENCHMARK_RESULT_PATH
from benchmark.utils_benchmark_case import BenchmarkCase
from benchmark.utils_data_transform_embeddings_by_pair import DataTransformEmbeddingsByPair
from benchmark.utils_data_extract_tar import DataExtractTarToFileDictionary
from benchmark.utils_data_load_embeddings_by_pair import DataLoadEmbeddingsByPair
from benchmark.utils_benchmark_runner_binarizer import BenchmarkRunnerBinarizer
from benchmark.utils_benchmark_runner_keygen import BenchmarRunnerkKeygen
from benchmark.utils_benchmark_report import BinarizerReport, KeygenReport
from lib.biocryp.binarizers import Static
from lib.biocryp.keygen import YimingFuzzyExtractor
import datetime


class CaseYiming(BenchmarkCase):
    def __init__(self, repeat: int = 500):
        binarizer = Static()
        keygen = YimingFuzzyExtractor(4096, 500, 128)

        current_datetime = f"{datetime.datetime.now()}"
        result_folder_path = f"{BENCHMARK_RESULT_PATH}/{current_datetime}"
        os.mkdir(result_folder_path)

        data_extract = DataExtractTarToFileDictionary(TAR_EMBEDDINGS_FILE_PATH)
        file_dictionary = data_extract.extract()
        data_transform = DataTransformEmbeddingsByPair(repeat, file_dictionary)
        data_load = DataLoadEmbeddingsByPair()
        binarizer_benchmark = BenchmarkRunnerBinarizer(binarizer)
        keygen_benchmark = BenchmarRunnerkKeygen(keygen)
        binarizer_report = BinarizerReport(result_folder_path)
        keygen_report = KeygenReport(result_folder_path)

        super().__init__(
            binarizer,
            keygen,
            turn_on_keygen_flag=True,
            n_samples=repeat,
            benchmark_transform=data_transform,
            data_loader=data_load,
            binarizer_benchmark=binarizer_benchmark,
            keygen_benchmark=keygen_benchmark,
            binarizer_report=binarizer_report,
            keygen_report=keygen_report,
        )


if __name__ == "__main__":
    """
    OK Yiming runs but extremely slow. Cannot really use that in practice. 
    The performance is also terrible (I don't know why).
    """
    CaseYiming(N_REPEAT).execute()
