import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))


from lib.camera import camera
from lib.feature import dnn
from lib.biocryp import binarizers
from client.constant import keygen_fuzzy_extractor, TEST_USER_ID, FEATURE_EXTRACTOR_ENFORCE_DETECTION_FLAG
from client.adapters import out_database_keyseed
import cv2


binarizer = binarizers.Static()


def execute(user_id: str):
    cv2_image_imread = camera.capture_to_cv2_single_image()
    cv2_image_imread = cv2.cvtColor(cv2_image_imread, cv2.COLOR_BGR2RGB)
    print("consent: done capturing.")

    feature_vector = dnn.extract_feature_vector(cv2_image_imread, enforce_detection=FEATURE_EXTRACTOR_ENFORCE_DETECTION_FLAG)
    binarized_feature_vector = binarizer.binarise(feature_vector)

    keyseed, ok = out_database_keyseed.get(user_id)

    if ok:
        result = keygen_fuzzy_extractor.verify(binarized_feature_vector, keyseed)

        if result == True:
            msg = "success"
        else:
            msg = "unauthorized unlock"
    else:
        result = False
        msg = "cannot find user"

    print("consent: result=", result, ", msg=", msg)

    return result, msg


if __name__ == "__main__":
    execute(TEST_USER_ID)