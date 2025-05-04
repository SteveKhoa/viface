"""
Some parts of the code was borrowed
"""

import abc
import numpy as np
from lib.biocryp import reed_solomon, contexts
from dataclasses import dataclass
from typing import List, Tuple
import bitstring
from pyldpc import make_ldpc, decode, get_message, utils


class ECC:
    """Abstract interface for binary (F2) error correcting codes."""

    # Some ECC is not systematic, therefore I cannot expose this interface
    # @abc.abstractmethod
    # def get_coding_matrix(self) -> np.ndarray:
    #     """Get systematric coding matrix `G = [I | A] transposed` of size `n, k` with `n > k`.

    #     To get `n`, `get_generator().shape[0]`.

    #     To get `k`, `get_generator().shape[1]`.
    #     """
    #     pass

    @abc.abstractmethod
    def get_coding_matrix_size(self):
        """Get size of coding matrix as `(n,k)`, where `n` is the length of codeword, `k` is the length of input matrix."""
        pass

    @abc.abstractmethod
    def encode(self, value: np.ndarray):
        pass

    @abc.abstractmethod
    def decode(self, value: np.ndarray, parity_codes: np.ndarray):
        """Error correction on `value` using `parity codes`"""
        pass


class ReedSolomon(ECC):
    """Reed Solomon error correcting code on GF(2^8), 256 symbols per block.

    Note: this is a workaround since other ECCs are difficult to implement.
    Through experiments I believe the length of input bitstring should <= 512 bits.
    """

    def __init__(self, message_length: int):
        """
        * message_length: bits per 255 bits (1 byte), parity bits will be `255 - message_length`.
        This is also known as `k` in theory.
        """

        self.rs = reed_solomon.RS(message_length)

    def get_coding_matrix_size(self):
        return (self.rs.n, self.rs.k)

    def encode(self, value: np.ndarray):
        """Encode np.ndarray of 0s and 1s to data and parity in 0s and 1s"""
        bitstring_ = contexts.BitString(value)
        pol: np.ndarray = np.frombuffer(
            bitstring_.as_bytes(),
            dtype=np.uint8,
        )  # map bitstring to GF(2^8)

        codeword = self.rs.encode(pol.tolist())
        data, parity = self.rs.data_parity_split(codeword)

        # To binary arrays
        data = np.array(data, dtype=np.uint8)
        data = np.unpackbits(data).tolist()
        parity = np.array(parity, dtype=np.uint8)
        parity = np.unpackbits(parity).tolist()
        return (data, parity)

    def decode(self, data, parity):
        """Error correction on `data` using `parity codes`"""

        # Transform arrays of 1s and 0s to GF(2^8) representation
        data = "".join(map(str, data))
        data = bitstring.ConstBitStream(bin=data)
        data = data.tobytes()
        data = np.frombuffer(data, dtype=np.uint8)
        parity = "".join(map(str, parity))
        parity = bitstring.ConstBitStream(bin=parity)
        parity = parity.tobytes()
        parity = np.frombuffer(parity, dtype=np.uint8)

        packet = data + parity
        decoded = self.rs.decode(packet)
        decoded = np.array(decoded, dtype=np.uint8)

        # Decoded packet returns data in GF(2^8), we have to map it back to binary field.
        org: List[int] = np.unpackbits(decoded).tolist()
        return org


class LDPC(ECC):
    """LDPC error correcting code."""

    def __init__(
        self,
        codeword_length: int = 320,
        d_v: int = 10,
        d_c: int = 16,
        snr: float = 8.192,
        maxiter: int = 100
    ):
        """
        * `codeword_length`: in bits
        * `d_v`: Number of parity-check equations including a certain bit.
        * `d_c`: Number of bits in the same parity-check equation. d_c Must be greater or equal to d_v and must divide n.
        * `snr`: signal to noise ratio (e.g. 4096 / 500 (total 4906, error 500) ~ 12.2% error)
        * `maxiter`: number of LDPC iterations
        """
        self.codeword_length = codeword_length
        self.d_v = d_v
        self.d_c = d_c
        self.snr = snr

        H, G = make_ldpc(codeword_length, d_v, d_c, systematic=True, sparse=True)
        self.H = H
        self.G = G

        self.data_length = G.shape[
            1
        ]  # also known as `k``. `k` cannot be determined deterministically, the only way is through empirical experiments

        self.maxiter = maxiter

    def _encode(self, v):
        """Rewritten from pyldpc's `encode`, because `_encode` returns non-noised codeword.

        Encode a binary message and adds Gaussian noise.

        Parameters
        ----------
        tG: array or scipy.sparse.csr_matrix (m, k). Transposed coding matrix
        obtained from `pyldpc.make_ldpc`.

        v: array (k, ) or (k, n_messages) binary messages to be encoded.

        snr: float. Signal-Noise Ratio. SNR = 10log(1 / variance) in decibels.

        Returns
        -------
        y: array (n,) or (n, n_messages) coded messages + noise.

        """
        d = utils.binaryproduct(self.G, v)
        x = (-1) ** d

        return x

    def get_coding_matrix_size(self):
        return self.G.shape

    def encode(self, value: np.ndarray):
        """Encode np.ndarray of 0s and 1s to data and parity in 0s and 1s"""

        value = value[: self.data_length]
        codeword = self._encode(value)

        data = codeword[: self.data_length]
        parity = codeword[self.data_length :]

        return (data, parity)

    def decode(self, value: np.ndarray, parity_codes: np.ndarray):
        """Error correction on `value` using `parity codes`"""

        new_codeword = np.concatenate([value, parity_codes])
        recovered_value = decode(self.H, new_codeword, snr=self.snr, maxiter=self.maxiter)

        recovered_data = get_message(self.G, recovered_value).tolist()
        return recovered_data


if __name__ == "__main__":
    pass
