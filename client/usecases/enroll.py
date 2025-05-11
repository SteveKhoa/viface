import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], "..", ".."))


from lib.camera import camera
from lib.feature import dnn
from lib.biocryp import binarizers
from client.constant import (
    keygen_fuzzy_extractor, 
    ENROLL_FROM_STATIC_DATA_DIR, 
    ENROLL_FROM_STATIC_DATA_FLAG, 
    TEST_USER_ID, 
    FEATURE_EXTRACTOR_DNN_MODEL, 
    FEATURE_EXTRACTOR_FACE_DETECTOR)
from client.adapters import out_database_keyseed
import cv2


binarizer = binarizers.Static()


def execute(user_id: str = "test_user") -> bool:
    if ENROLL_FROM_STATIC_DATA_FLAG == "True":
        path_list = os.listdir(ENROLL_FROM_STATIC_DATA_DIR)
        user_id = TEST_USER_ID

        cv2_image_imreads = [cv2.imread(os.path.join(ENROLL_FROM_STATIC_DATA_DIR, path)) for path in path_list]
    else:
        cv2_image_imreads = camera.capture_to_cv2_multiple_images()

    cv2_image_imreads = [cv2.cvtColor(cv2_image_imread, cv2.COLOR_BGR2RGB) for cv2_image_imread in cv2_image_imreads]

    print("enroll: model=", FEATURE_EXTRACTOR_DNN_MODEL)
    print("enroll: face detector=", FEATURE_EXTRACTOR_FACE_DETECTOR)

    feature_vectors = []
    for im in cv2_image_imreads:
        feature_vector, ok = dnn.extract_feature_vector(im, FEATURE_EXTRACTOR_DNN_MODEL, FEATURE_EXTRACTOR_FACE_DETECTOR)

        if not ok:
            print("enroll: face cannot be detected... skip.")
            continue

        feature_vectors += [feature_vector]

    binarized_feature_vectors = [binarizer.binarise(feat) for feat in feature_vectors]

    concatenated_binarized_feature_vectors = b"".join(binarized_feature_vectors)

    _, keyseed = keygen_fuzzy_extractor.generate(concatenated_binarized_feature_vectors)

    out_database_keyseed.save(keyseed, user_id)

    return True


if __name__ == "__main__":
    if ENROLL_FROM_STATIC_DATA_FLAG == "True":
        execute()