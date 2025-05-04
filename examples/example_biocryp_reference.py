from lib.biocryp import binarizers, keygen
import feature
from PIL import Image
import sys
import pickle
import tempfile

if __name__ == "__main__":
    pil_image = Image.open(f"./lfw/Zico_0003.jpg")

    # Face preprocessing
    embedding = feature.to_embedding(pil_image)
    meta = binarizers.DROBA.Meta().load()
    binarizer_ = binarizers.DROBA(meta)
    bitstring = binarizer_.binarise(embedding)

    # Encryption
    fuzzy_extractor = keygen.DefaultKeygen()
    key, helper = fuzzy_extractor._generate(bitstring)
    login_key = fuzzy_extractor._reproduce(bitstring, helper)

    # Print the size of `helper` in bytes, useful to monitor helper string size.
    # https://stackoverflow.com/questions/49934598/how-can-i-get-the-size-of-a-temporaryfile-in-python
    with tempfile.SpooledTemporaryFile() as write_buf:
        pickle.dump(helper, write_buf, pickle.HIGHEST_PROTOCOL)
        print(write_buf.tell())

    # Login with the same image, should be equivalent key.
    print(key)
    print(login_key)
    print(key == login_key)