# The way I organize code into this file is terribly wrong, planned to fix this later.

# Binarizer + Fuzzy Extractor
BITLENGTH = 4096
BYTE_LENGTH = int(BITLENGTH / 8)

# Fuzzy Extractor specific
HAMMING_THRESHOLD = 5  # more hamming, more time required
REP_ERR = 0.00000001  # less rep_err, more time required
HASH_FUNC = "sha256"
SECURITY_LENGTH = 80  # More SECURITY_LENGTH will require more storage size, but performance is not affected. Also called Security Parameter, in this paper A Reusable Fuzzy Extractor with Practical Storage Size: Modifying Canetti et al.’s Construction, it is suggested to use the number 80
NDIMS = 512
BINARIZER_PATH = "./binarizer.pickle"
HELPERS_PATH = "helpers.pickle"

# UI
USERNAME = "stevekhoa"