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
    keyseed, ok = out_database_keyseed.get(user_id)

    if not ok:
        result = False
        msg = "cannot find user"
        print("consent: result=", result, ", msg=", msg)

        return result, msg

    cv2_image_imread = camera.capture_to_cv2_single_image()
    cv2_image_imread = cv2.cvtColor(cv2_image_imread, cv2.COLOR_BGR2RGB)
    print("consent: done capturing.")

    print("consent: enforce_detection=", FEATURE_EXTRACTOR_ENFORCE_DETECTION_FLAG)
    feature_vector, ok = dnn.extract_feature_vector(cv2_image_imread, enforce_detection=FEATURE_EXTRACTOR_ENFORCE_DETECTION_FLAG)

    if not ok:
        result = False
        msg = "face cannot be detected"
        print("consent: result=", result, ", msg=", msg)

        return result, msg

    binarized_feature_vector = binarizer.binarise(feature_vector)

    result = keygen_fuzzy_extractor.verify(binarized_feature_vector, keyseed)

    if result == True:
        msg = "success"
    else:
        msg = "unauthorized sign-in"

    print("consent: result=", result, ", msg=", msg)

    return result, msg


if __name__ == "__main__":
    execute(TEST_USER_ID)