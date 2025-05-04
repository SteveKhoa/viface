import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np
import pickle
from lib.biocryp import keygen, vars, binarizers


class QuantizationCoding:
    """
    For compatibility with old binarizer.pickle.
    There is no practical purpose at all.
    """
    @staticmethod
    def generate_plssc(*args, **kwargs):
        pass


class Biocryp:
    """Interface for Biometric Encryption.
    """
    @staticmethod
    def _preprocess(face_features: np.ndarray):
        binarizer = binarizers.Static()
        bitstring = binarizer.binarise(face_features)
        return bitstring

    @staticmethod
    def encrypt_and_save(face_features: np.ndarray) -> bytes:
        """Encrypt input biometrics and save encrypted stream to disk
        """
        bitstring = Biocryp._preprocess(face_features)

        fuzz = keygen.DefaultKeygen()
        key, helper = fuzz._generate(bitstring)

        datastructure = {
            "userid": "ASFA-234234-SDAFASDF",
            "username": vars.USERNAME,
            "helper": helper,
        }

        with open(vars.HELPERS_PATH, "wb") as write_f:
            pickle.dump(datastructure, write_f, pickle.HIGHEST_PROTOCOL)
        return key

    @staticmethod
    def load_and_decrypt(face: np.ndarray) -> bytes:
        """Load saved encryption and reproduce cryptographic key
        """
        bitstring = Biocryp._preprocess(face)

        with open(vars.HELPERS_PATH, "rb") as read_f:
            datastructure = pickle.load(read_f)
            helper = datastructure["helper"]

        fuzz = keygen.DefaultKeygen()
        key = fuzz._reproduce(bitstring, helper)
        return key
