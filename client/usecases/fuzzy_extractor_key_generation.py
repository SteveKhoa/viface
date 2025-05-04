from typing import List
import json
from lib.biocryp import binarizers
from client import constant
import base64
import numpy as np


def execute_key_generation(feature_vectors: List[np.ndarray], key_id: str) -> bytes:
    binarizer_static = binarizers.Static()

    binarized_feature_vectors = b""
    for feature_vector in feature_vectors:
        binarized_feature_vector = binarizer_static.binarise(feature_vector)
        binarized_feature_vectors += binarized_feature_vector

    (
        key,
        (
            cipher,
            mask,
            nonce,
        ),
    ) = constant.keygen_fuzzy_extractor._generate(binarized_feature_vectors)

    helper_path = f"{constant.BIOCRYP_HELPER_DIR}/biocryp_helper_{key_id}_lastest.json"
    with open(helper_path, "w") as f:
        cipher_b32 = base64.b32encode(cipher).decode("ascii")
        mask_b32 = base64.b32encode(mask).decode("ascii")
        nonce_b32 = base64.b32encode(nonce).decode("ascii")

        data = {
            "cipher_b32": cipher_b32,
            "mask_b32": mask_b32,
            "nonce_b32": nonce_b32,
        }
        json.dump(data, f)

    return key


if __name__ == "__main__":
    pass
