import scipy.special
from typing import List
from lib.biocryp import ecc, vars, hash, contexts
from math import log
from os import urandom
from fastpbkdf2 import pbkdf2_hmac
import numpy as np
from typing import Any, Tuple, Optional, Union
import abc
import bitstring
import scipy
import random
import json
import hmac
import numpy as np
import random
import string
import time
from hashlib import sha512
import warnings

Keyseed = Tuple[np.ndarray, np.ndarray, np.ndarray]
CryptographicKey = bytes


class BaseKeygen(object):
    @abc.abstractmethod
    def info(self) -> dict:
        """Returns meta information such as length_bitstring"""
        pass

    @abc.abstractmethod
    def generate(
        self,
        biometric: bytes,
        keyseed: Optional[Keyseed] = None,
    ) -> Tuple[CryptographicKey, Keyseed]:
        """Generate a cryptographic key from biometric. To reproduce predictable keys, provide previously stored `keyseed`. Otherwise, generated key will be unpredictable"""
        pass


class CanettiFuzzyExtractor(BaseKeygen):
    """The most basic non-interactive fuzzy extractor

    GPL-3.0 LICENSE: https://github.com/carter-yagemann/python-fuzzy-extractor
    """

    def __init__(self, length, ham_err, rep_err=0.001, **locker_args):
        """Initializes a fuzzy extractor

        :param length: The length in bytes of source values and keys.
        :param ham_err: Hamming error. The number of bits that can be flipped in the
            source value and still produce the same key with probability (1 - rep_err).
        :param rep_err: Reproduce error. The probability that a source value within
            ham_err will not produce the same key (default: 0.001).
        :param locker_args: Keyword arguments to pass to the underlying digital lockers.
            See parse_locker_args() for more details.
        """
        self.key_length = int(128 / 8)

        self._parse_locker_args(**locker_args)
        self.length = length
        self.cipher_len = self.key_length + self.sec_len

        # Calculate the number of helper values needed to be able to reproduce
        # keys given ham_err and rep_err. See "Reusable Fuzzy Extractors for
        # Low-Entropy Distributions" by Canetti, et al. for details.
        bits = length * 8
        const = float(ham_err) / log(bits)
        num_helpers = (bits**const) * log(float(2) / rep_err, 2)

        # num_helpers needs to be an integer
        self.num_helpers = int(round(num_helpers))

    def _parse_locker_args(self, hash_func="sha256", sec_len=2, nonce_len=16):
        """Parse arguments for digital lockers

        :param hash_func: The hash function to use for the digital locker (default: sha256).
        :param sec_len: security parameter. This is used to determine if the locker
            is unlocked successfully with accuracy (1 - 2 ^ -sec_len).
        :param nonce_len: Length in bytes of nonce (salt) used in digital locker (default: 16).
        """
        self.hash_func = hash_func
        self.sec_len = sec_len
        self.nonce_len = nonce_len

    def _generate(self, value):
        """Takes a source value and produces a key and public helper

        This method should be used once at enrollment.

        Note that the "public helper" is actually a tuple. This whole tuple should be
        passed as the helpers argument to reproduce().

        :param value: the value to generate a key and public helper for.
        :rtype: (key, helper)
        """
        if isinstance(value, (bytes, str)):
            value = np.fromstring(value, dtype=np.uint8)

        key = np.fromstring(urandom(self.key_length), dtype=np.uint8)
        key_pad = np.concatenate((key, np.zeros(self.sec_len, dtype=np.uint8)))

        nonces = np.zeros((self.num_helpers, self.nonce_len), dtype=np.uint8)
        masks = np.zeros((self.num_helpers, self.length), dtype=np.uint8)
        digests = np.zeros((self.num_helpers, self.cipher_len), dtype=np.uint8)

        # Config sampler
        nbits = self.length * 8
        k_const = 64  # number of bits expected to be sampled
        prob = k_const / float(nbits)

        for helper in range(self.num_helpers):
            nonces[helper] = np.fromstring(urandom(self.nonce_len), dtype=np.uint8)

            sample_indexes = np.random.binomial(1, prob, nbits)
            binstringmask = "".join(map(str, sample_indexes))
            bitstring_object = bitstring.ConstBitStream(bin=binstringmask)
            masks[helper] = np.fromstring(bitstring_object.tobytes(), dtype=np.uint8)

            # masks[helper] = np.fromstring(urandom(self.length), dtype=np.uint8)

        # By masking the value with random masks, we adjust the probability that given
        # another noisy reading of the same source, enough bits will match for the new
        # reading & mask to equal the old reading & mask.

        vectors = np.bitwise_and(masks, value)

        # The "digital locker" is a simple cyrpto primitive made by hashing a "key"
        # xor a "value". The only efficient way to get the value back is to know
        # the key, which can then be hashed again xor the ciphertext. This is referred
        # to as locking and unlocking the digital locker, respectively.

        for helper in range(self.num_helpers):
            d_vector = vectors[helper].tobytes()
            d_nonce = nonces[helper].tobytes()
            digest = pbkdf2_hmac(self.hash_func, d_vector, d_nonce, 1, self.cipher_len)
            digests[helper] = np.fromstring(digest, dtype=np.uint8)

        ciphers = np.bitwise_xor(digests, key_pad)

        return (key.tobytes(), (ciphers, masks, nonces))

    def _reproduce(self, value, helpers):
        """Takes a source value and a public helper and produces a key

        Given a helper value that matches and a source value that is close to
        those produced by generate, the same key will be produced.

        :param value: the value to reproduce a key for.
        :param helpers: the previously generated public helper.
        :rtype: key or None
        """
        if isinstance(value, (bytes, str)):
            value = np.fromstring(value, dtype=np.uint8)

        if self.length != len(value):
            raise ValueError("Cannot reproduce key for value of different length")

        ciphers = helpers[0]
        masks = helpers[1]
        nonces = helpers[2]

        vectors = np.bitwise_and(masks, value)

        digests = np.zeros((self.num_helpers, self.cipher_len), dtype=np.uint8)
        for helper in range(self.num_helpers):
            d_vector = vectors[helper].tobytes()
            d_nonce = nonces[helper].tobytes()
            digest = pbkdf2_hmac(self.hash_func, d_vector, d_nonce, 1, self.cipher_len)
            digests[helper] = np.fromstring(digest, dtype=np.uint8)

        plains = np.bitwise_xor(digests, ciphers)

        # When the key was stored in the digital lockers, extra null bytes were added
        # onto the end, which makes it each to detect if we've successfully unlocked
        # the locker.

        checks = np.sum(plains[:, -self.sec_len :], axis=1)
        for check in range(self.num_helpers):
            if checks[check] == 0:
                return plains[check, : -self.sec_len].tobytes()

        return None

    def info(self):
        return {"length_bitstring": self.length * 8}

    def generate(
        self,
        biometric: bytes,
        keyseed: Optional[Keyseed] = None,
    ) -> Tuple[CryptographicKey, Keyseed]:
        """Generate a cryptographic key from biometric. To reproduce predictable keys, provide previously stored `keyseed`"""
        if keyseed is None:
            (key, keyseed) = self._generate(biometric)
        else:
            key = self._reproduce(biometric, keyseed)
            if key is None:
                key = urandom(int(self.key_length / 8))
        return (key, keyseed)


class MaskThenLockFuzzyExtractor(BaseKeygen):
    """My proposal on a construction of fuzzy extractor.

    Other fuzzy extractor constructions are either difficult to implement, or does not achieve good performance.

    In this construction, I assume that multiple input biometrics (at registration) of the same person must be
    provided. This helps the fuzzy extractor to learns useful information about one person, from there, can create
    better masks, instead of randomly sample from the original string as in Canetti's approach.

    My construction is very similar to Canetti's, the only difference is that the mask is tailored to each person,
    to create better samples as input to the digital locker.
    """

    def __init__(
        self,
        input_length: int,
        key_length: int,
        mask_length: int,
        nonce_len: int,
        sec_len: int = 8,
        hash_func="sha256",
    ):
        """
        * input_length: input length in bytes
        * key_length: key lenth in bytes
        * sec_len: security parameter of the digital locker, in bytes
        * nonce_len: length of nonce in bytes

        * mask_length: length of entropy mask in bits
        """
        self.input_length = input_length
        self.key_length = key_length
        self.sec_len = sec_len
        self.cipher_len = self.key_length + sec_len
        self.hash_func = hash_func
        self.nonce_len = nonce_len
        self.mask_length = mask_length
        assert self.mask_length < self.input_length * 8

        # Registration index is the index of the sample which will be registered
        # among many input samples (values).
        self.registration_index = 0

        self.log_err = (
            0.00000001  # Tolerable error to fix `log2(0)` in entropy computation
        )

    def info(self):
        return {"length_bitstring": self.input_length * 8}

    def _generate(self, values: List[bytes]):
        """
        We demand many values as variants of the same person, so that we can build the mask.
        `values` is the array of bytes of length `n = input_length * number_of_values`.
        """

        # Parse bytes by cutting segments into separate BitString context
        n_values = int(len(values) / self.input_length)
        assert n_values >= 2
        binstringlist = []
        for i in range(n_values):
            start = int(i * self.input_length)
            end = int(start + self.input_length)
            sliced = values[start:end]
            binstringlist += [contexts.BitString(sliced).as_nparray().tolist()]

        # Compute entropy for each bit in `values`
        np_ = np.array(binstringlist)
        sum_ = np_.sum(axis=0, keepdims=True)
        p = sum_ / np_.shape[0]
        entropies = -((p + self.log_err) * np.log2(p + self.log_err)) - (
            (1.0 - p + self.log_err) * np.log2(1.0 - p + self.log_err)
        )
        entropies = np.squeeze(entropies).tolist()

        # Create mask according to entropy
        sorted_args = np.argsort(np.array(entropies))
        sorted_args = sorted_args[: self.mask_length]
        mask = np.zeros(int(self.input_length * 8), dtype=np.uint8)
        mask[sorted_args] = 1
        mask = contexts.BitString(np.array(mask)).as_bytes()
        mask = np.frombuffer(mask, dtype=np.uint8)

        # Generate random uniform key
        key: np.ndarray = np.frombuffer(urandom(self.key_length), dtype=np.uint8)
        key_pad = np.concatenate((key, np.zeros(self.sec_len, dtype=np.uint8)))

        # Choose one value as the baseline
        value = binstringlist[self.registration_index]
        value = contexts.BitString(value).as_bytes()
        value = np.frombuffer(value, dtype=np.uint8)

        # Mask the vectors
        vector: np.ndarray = np.bitwise_and(value, mask)

        # print(vector)

        # Lock using digital locker
        nonce: np.ndarray = np.frombuffer(urandom(self.nonce_len), dtype=np.uint8)
        vector = vector.tobytes()
        nonce = nonce.tobytes()
        digest = pbkdf2_hmac(self.hash_func, vector, nonce, 1, self.cipher_len)
        digest = np.fromstring(digest, dtype=np.uint8)
        # print(digest)
        cipher = np.bitwise_xor(digest, key_pad)

        cipher: bytes = cipher.tobytes()
        mask: bytes = mask.tobytes()

        # print(nonce)

        return (key.tobytes(), (cipher, mask, nonce))

    def _reproduce(self, value: bytes, helpers: Tuple):
        (cipher, mask, nonce) = helpers

        cipher = np.frombuffer(cipher, dtype=np.uint8)
        value = np.frombuffer(value, dtype=np.uint8)
        mask = np.frombuffer(mask, dtype=np.uint8)

        vector: np.ndarray = np.bitwise_and(value, mask)

        # print(vector)

        # print(nonce)

        # Unlock using digital locker
        vector = vector.tobytes()
        digest = pbkdf2_hmac(self.hash_func, vector, nonce, 1, self.cipher_len)
        digest = np.fromstring(digest, dtype=np.uint8)
        # print(digest)
        plain: np.ndarray = np.bitwise_xor(digest, cipher)

        # When the key was stored in the digital lockers, extra null bytes were added
        # onto the end, which makes it each to detect if we've successfully unlocked
        # the locker.

        check = np.sum(plain[-self.sec_len :])
        # print(plain)
        # print(plain[-self.sec_len :])
        # print()
        if check == 0:
            reproduced_key: np.ndarray = plain[: -self.sec_len]
            return reproduced_key.tobytes()
        else:
            return None

    def generate(
        self,
        biometric: bytes,
        keyseed: Optional[Keyseed] = None,
    ) -> Tuple[CryptographicKey, Keyseed]:
        """Generate a cryptographic key from biometric. To reproduce predictable keys, provide previously stored `keyseed`"""
        if keyseed is None:
            (key, keyseed) = self._generate(biometric)
        else:
            key = self._reproduce(biometric, keyseed)
            if key is None:
                key = urandom(self.key_length)
        return (key, keyseed)

    def verify(
        self,
        biometric: bytes,
        keyseed: Keyseed,
    ) -> bool:
        key = self._reproduce(biometric, keyseed)

        if key is None:
            return False
        else:
            return True


# This doesn't work I don't know why. There's something with the error correcting codes
# in this construction. In short, this approach is just too hard for me.
#
class YimingFuzzyExtractor(BaseKeygen):
    """Reusable Fuzzy Extractor Based on the LPN Assumption (2020)"""

    def __init__(self, input_length: int, ham_err: int = None, keylength: int = None):
        """
        * input_length: number of bits in input bit string
        * ham_err: correctable hamming errors (number of bits)
        * keylength: length of key in bits
        """

        if input_length is not None:
            warnings.warn(
                "input_length is not controllable right now.... I did not manage to fully control ECC according to `input_length`"
            )

        if ham_err is not None:
            warnings.warn(
                "ham_err is not controllable right now.... I did not manage to fully control ECC according to `ham_err`"
            )

        if keylength is not None:
            warnings.warn(
                "keylength is not controllable right now.... I did not manage to fully control ECC according to `keylength`"
            )

        self.ecc1: ecc.ECC = ecc.LDPC(10000, maxiter=1000)
        self.ecc2: ecc.ECC = ecc.LDPC(352, maxiter=1000)
        self.keylength = self.ecc2.get_coding_matrix_size()[1]
        self.input_length = self.ecc1.get_coding_matrix_size()[1]

        self.hash_func: hash.UniversalHash = hash.UniversalHash(
            self.input_length,
            self.input_length,
        )  # "the sun can only count up to about 190-200 bits"

        # Configuration constants
        self.GALOIS_ORDER = 2  # GF(2^m)
        self.BIONOMIAL_DIST_PROB = 0.7  # magic number!

    def _generate(self, biometric: bytes):
        inp: contexts.BitString = contexts.BitString(biometric)
        inp_binlist = inp.as_nparray().tolist()

        # Pad the input with zeros if input is smaller size compared to input length
        if len(inp_binlist) < self.input_length:
            pad = [0] * (self.input_length - len(inp_binlist))
            inp_binlist = inp_binlist + pad

        _, parity_codes = self.ecc1.encode(np.array(inp_binlist))

        # Learning parity with noise (LPN)
        # https://medium.com/zkpass/learning-parity-with-noise-lpn-55450fd4969c
        #
        # As + e mod p
        inp = contexts.BitString(np.array(inp_binlist))
        s = self.hash_func.hash(inp)
        s = s.as_nparray()
        s_length = self.hash_func.output_length
        n2, k2 = self.ecc2.get_coding_matrix_size()
        A = np.random.randint(self.GALOIS_ORDER, size=(n2, s_length))
        key = np.random.randint(self.GALOIS_ORDER, size=k2)
        e = np.random.binomial(1, self.BIONOMIAL_DIST_PROB, n2)
        key_padded, key_parity = self.ecc2.encode(key)
        key_encoded = np.concatenate([key_padded, key_parity])

        lpn_challenge = key_encoded.T + A @ s + e.T

        return (key, (parity_codes, A, lpn_challenge))

    def _reproduce(self, biometric: bytes, helper):
        inp: contexts.BitString = contexts.BitString(biometric)
        inp_binlist = inp.as_nparray()
        # Pad the input with zeros if input is smaller size compared to input length
        if len(inp_binlist) < self.input_length:
            inp_binlist += [0] * (self.input_length - len(inp_binlist))

        (parity_codes, A, lpn_challenge) = helper

        augmented = (-1) ** inp_binlist

        # Login biometric is never the same as enrolled biometric,
        # therefore we must perform error correction.
        recovered_biometric = self.ecc1.decode(np.array(augmented), parity_codes)
        recovered_biometric = contexts.BitString(np.array(recovered_biometric))

        # Solve LPN challenge to recover the key
        s_tilde = self.hash_func.hash(recovered_biometric)
        pack = lpn_challenge - A @ np.array(s_tilde.as_nparray())
        _, k2 = self.ecc2.get_coding_matrix_size()
        recovered_key = self.ecc2.decode(pack[:k2], pack[k2:])

        recovered_key = contexts.BitString(recovered_key).as_bytes()

        return recovered_key

    def info(self) -> dict:
        """Returns meta information such as length_bitstring"""
        return {"length_bitstring": self.input_length}

    def generate(
        self,
        biometric: bytes,
        keyseed: Optional[Keyseed] = None,
    ) -> Tuple[CryptographicKey, Keyseed]:
        """Generate a cryptographic key from biometric. To reproduce predictable keys, provide previously stored `keyseed`. Otherwise, generated key will be unpredictable"""

        if keyseed is None:
            key, keyseed = self._generate(biometric)
        else:
            key = self._reproduce(biometric, keyseed)
            if key is None:
                key = urandom(self.key_length)
        return (key, keyseed)


class DefaultKeygen(CanettiFuzzyExtractor):
    """Interface for Key Generator"""

    def __init__(self):
        super().__init__(
            vars.BYTE_LENGTH,
            vars.HAMMING_THRESHOLD,
            rep_err=vars.REP_ERR,
            hash_func=vars.HASH_FUNC,
            sec_len=vars.SECURITY_LENGTH,
        )


if __name__ == "__main__":
    pass
