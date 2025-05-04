from typing import List
import json
from lib.biocryp import binarizers
from client import constant
import base64
import numpy as np


def execute_key_reproduction(feature_vector: np.ndarray, key_id: str):
    binarizer_static = binarizers.Static()
    binarized_feature_vector = binarizer_static.binarise(feature_vector)

    helper_path = f"{constant.BIOCRYP_HELPER_DIR}/biocryp_helper_{key_id}_lastest.json"
    with open(helper_path, "r") as f:
        data = json.load(f)

        cipher_b32 = data["cipher_b32"]
        mask_b32 = data["mask_b32"]
        nonce_b32 = data["nonce_b32"]

        cipher = base64.b32decode(cipher_b32)
        mask = base64.b32decode(mask_b32)
        nonce = base64.b32decode(nonce_b32)

    key = constant.keygen_fuzzy_extractor._reproduce(
        binarized_feature_vector,
        (cipher, mask, nonce),
    )

    return key


if __name__ == "__main__":
    pass
