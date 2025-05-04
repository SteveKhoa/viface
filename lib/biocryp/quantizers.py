import numpy as np
import math
import abc

EPSILON_ADJUSMENT = 0.0000001


class BaseQuantizer(object):
    @abc.abstractmethod
    def quantize(self, value: float, min_value: float, max_value: float) -> int:
        pass


class AnyQuantizer(BaseQuantizer):
    """
    Turn individual real value to a binary label.

    This is a crucial step in binarization. Note, this always use equal-width
    intervals (instead of equal-probable).
    """

    def __init__(self, coding_scheme="lssc", nbits=2):
        if coding_scheme == "lssc":
            self.coding_table, self.n_intervals = AnyQuantizer._generate_lssc(nbits)
        elif coding_scheme == "plssc":
            self.coding_table, self.n_intervals = AnyQuantizer._generate_plssc(nbits)
        elif coding_scheme == "brgc":
            self.coding_table, self.n_intervals = AnyQuantizer._generate_brgc(nbits)
        else:
            raise ValueError("invalid coding_scheme")

    @staticmethod
    def _generate_brgc(length):
        """
        Build Binary Reflected Gray Code (BRGC), represented as integer
        https://stackoverflow.com/questions/38738835/generating-gray-codes-recursively-without-initialising-the-base-case-outside-of
        """
        codelength = 1 << length
        gray_codes = []
        for i in range(0, 1 << length):
            gray = i ^ (i >> 1)
            gray_codes += [gray]
            # print("{0:0{1}b}".format(gray,length), ' --- ', gray)
        return np.array(gray_codes), codelength

    @staticmethod
    def _generate_lssc(length):
        codelength = length
        lssc_lst = [0]
        for i in range(1, length + 1):
            lssc = lssc_lst[i - 1] | (1 << (i - 1))
            lssc_lst += [lssc]
        # [print("{0:0{1}b}".format(lssc,length), ' --- ', lssc) for lssc in lssc_lst]
        return np.array(lssc_lst), codelength + 1

    @staticmethod
    def _generate_plssc(length):
        codelength = length
        lssc_lst = [0]
        for i in range(1, length + 1):
            lssc = lssc_lst[i - 1] | (1 << (i - 1))
            lssc_lst += [lssc]
        tail = []
        for i in range(1, length):
            lssc = ~lssc_lst[i] + (1 << length)
            tail += [lssc]
        lssc_lst = lssc_lst + tail
        # [print("{0:0{1}b}".format(gray, length), " --- ", gray) for gray in lssc_lst]
        return np.array(lssc_lst, dtype=np.longlong), 2 * codelength

    def quantize(self, value: float, min_value: float, max_value: float) -> int:
        """
        Get individual binary label for `value` using my elegant algorithm ^^
        """
        incremental_value = (max_value - min_value) / (self.n_intervals)
        if value < min_value:
            i = 0
        elif value >= max_value:
            i = self.coding_table.shape[0] - 1
        else:
            i = math.floor((value - EPSILON_ADJUSMENT - min_value) / incremental_value)
        return self.coding_table[i]
