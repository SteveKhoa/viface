import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from benchmark.constant import N_REPEAT, TAR_EMBEDDINGS_FILE_PATH, BENCHMARK_RESULT_PATH
from benchmark.utils_benchmark_case import BenchmarkCase
from benchmark.utils_data_transform_embeddings_by_pair import (
    DataTransformEmbeddingsByPair,
)
from benchmark.utils_data_extract_tar import DataExtractTarToFileDictionary
from benchmark.utils_data_load_embeddings_by_pair import DataLoadEmbeddingsByPair
from lib.biocryp.binarizers import Static

import numpy as np
from numpy import dot
from numpy.linalg import norm

INVERSE_COSINE_SIMILARITY_THRESHOLD = -0.3
HAMMING_THRESHOLD = 450


def main(inverse_cosine_similarity_threshold: float, hamming_threshold: int, repeat: int = 500):
    data_extract = DataExtractTarToFileDictionary(TAR_EMBEDDINGS_FILE_PATH)
    file_dictionary = data_extract.extract()
    data_transform = DataTransformEmbeddingsByPair(repeat, file_dictionary)
    data_loader = DataLoadEmbeddingsByPair()
    binarizer = Static()

    people_struct_pair_list = data_transform.transform()
    len_people_struct_pair_list = float(len(people_struct_pair_list))

    print("DNN Match Rate: transforming dataset, done.")

    dnn_total_false_match = 0.0
    dnn_total_false_notmatch = 0.0
    binarization_total_false_match = 0.0
    binarization_total_false_notmatch = 0.0

    cos_sim = lambda a, b: dot(a, b) / (norm(a) * norm(b))

    for _, people_struct_pair in enumerate(people_struct_pair_list):
        (
            base_signup_np_emb,
            base_login_np_emb,
            false_login_np_emb,
        ) = data_loader.load_one_sample(people_struct_pair)

        same_distance = -abs(cos_sim(base_signup_np_emb, base_login_np_emb))
        diff_distance = -abs(cos_sim(base_signup_np_emb, false_login_np_emb))
        dnn_total_false_match += (diff_distance < inverse_cosine_similarity_threshold)
        dnn_total_false_notmatch += (same_distance > inverse_cosine_similarity_threshold)


        base_signup_np_boolarr = binarizer.binarise_asbool(base_signup_np_emb)
        base_login_np_boolarr = binarizer.binarise_asbool(base_login_np_emb)
        false_login_np_boolarr = binarizer.binarise_asbool(false_login_np_emb)

        same_distance = np.count_nonzero(base_signup_np_boolarr != base_login_np_boolarr)
        diff_distance = np.count_nonzero(base_signup_np_boolarr != false_login_np_boolarr)
        binarization_total_false_match += (diff_distance < hamming_threshold)
        binarization_total_false_notmatch += (same_distance > hamming_threshold)



    print("DNN_FMR =", dnn_total_false_match / len_people_struct_pair_list)
    print("DNN_FNMR =", dnn_total_false_notmatch / len_people_struct_pair_list)
    print("BINARIZATION_FMR =", binarization_total_false_match / len_people_struct_pair_list)
    print("BINARIZATION_FNMR =", binarization_total_false_notmatch / len_people_struct_pair_list)
    


if __name__ == "__main__":
    main(INVERSE_COSINE_SIMILARITY_THRESHOLD, HAMMING_THRESHOLD, N_REPEAT)
