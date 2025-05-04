"""
Miscellaneous files
"""

def to_bytestring(bool_arr, length):
    x = sum(map(lambda x: x[1] << x[0], enumerate(bool_arr)))
    bin_ = int(x).to_bytes(length=length, byteorder="big", signed=True)
    return bin_


def fib_list(n):
    if n <= 0:
        return []
    if n == 1:
        return [0]

    result = [0, 1]
    while len(result) < n:
        result.append(result[-1] + result[-2])
    return result