import numpy as np
from typing import Union
import bitstring


class BitString:
    """Context object, abstraction over `bytes` and `np.ndarray`
    to mitigate processing ambiguity between boolarr and bytes.
    """

    def __init__(self, value: Union[bytes, np.ndarray]):
        if isinstance(value, bytes):
            self.bitstream: bitstring.ConstBitStream = bitstring.ConstBitStream(value)
        elif isinstance(value, np.ndarray) or isinstance(value, list):
            binarystring = "".join(map(str, value))
            self.bitstream: bitstring.ConstBitStream = bitstring.ConstBitStream(
                bin=binarystring
            )
        else:
            raise Exception("unexpected data type:", type(value))

    def as_nparray(self):
        """List of 0s and 1s"""

        lst = self.bitstream.tobitarray().tolist()
        return np.array(lst)

    def as_bytes(self):
        """Python byte reprensetation"""
        return self.bitstream.tobytes()