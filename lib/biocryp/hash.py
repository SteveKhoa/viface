import numpy as np
from lib.biocryp import contexts


class UniversalHash:
    """Linear algebra-based hash.

    See "Universal hashing" in Yiming Li et al. (2020)

    * input_length:   number of input bits
    * output_length:  number of output bits
    """

    def __init__(self, input_length: int, output_length: int):
        self.input_length = input_length
        self.output_length = output_length

        # Continuously sample until we got a l-ranked random matrix.
        # Stop at STOP_THRESHOLD to prevent stack overflow.
        STOP_THRESHOLD = 100
        self.hash_seed = None
        for i in range(STOP_THRESHOLD):
            A = np.random.randint(
                0,
                2,
                size=(output_length, input_length),
                dtype=np.uint8,
            )
            if np.linalg.matrix_rank(A) == output_length:
                break
        else:
            raise Exception("Cannot find a l-ranked matrix")
        self.hash_seed = A

    def hash(self, input: contexts.BitString) -> contexts.BitString:
        """Produce a bitstring (1s and 0s) of length `self.output_length`"""
        input = input.as_nparray()
        assert len(input.shape) == 1  # assert input is a column vector
        input_length = input.shape[0]

        print("input_length=", input_length)
        print("self.input_length=", self.input_length)
        assert input_length == self.input_length  # make sure input length matched the design

        digest = self.hash_seed.dot(input) % 2  # make sure it is in binary domain
        return contexts.BitString(digest)
