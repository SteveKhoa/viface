import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from random import randint
from lib.feature import dnn, augmentation
from benchmark import utils_time_measurement
from sklearn.preprocessing import normalize
import numpy as np


class DataLoadImagesByPairForMaskThenLock:
    def _split_true_and_false_embeddings(self, people_struct_pair):
        # Get a true identity and a false identity
        true_identity = 0
        false_identity = 1

        # Acquire base
        base_images = people_struct_pair[true_identity][-1]["images"]
        base_length = people_struct_pair[true_identity][-1]["n_images"]

        # Acquire false
        false_images = people_struct_pair[false_identity][-1]["images"]
        false_length = people_struct_pair[false_identity][-1]["n_images"]

        return (
            (base_images, base_length),
            (false_images, false_length),
        )

    def _divide_into_register_login_samples(
        self,
        base_images,
        base_length,
        false_images,
        false_length,
    ):
        # Select two random samples in reference (base) identity
        base_select_signup = randint(0, base_length - 1)
        base_select_login = randint(0, base_length - 1)
        while (
            base_select_login == base_select_signup
        ):  # make sure two samples are not the same
            base_select_login = randint(0, base_length - 1)

        base_signup_np_image = base_images[base_select_signup]
        base_login_np_image = base_images[base_select_login]

        # Select one random sample from false identity
        false_select_login = randint(0, false_length - 1)
        false_login_np_image = false_images[false_select_login]

        return (
            base_signup_np_image,
            base_login_np_image,
            false_login_np_image,
        )
    

    @utils_time_measurement.decorator_time_measurement_log("DataLoadImagesByPair: image augmentation + feature extraction, done.")
    def _do_image_augmentation(self, base_signup_np_image, base_login_np_image, false_login_np_image):
        augmented_base_signup_np_images = augmentation.create_augmented_images(base_signup_np_image)

        # Feature extraction
        base_signup_np_embs = [dnn.extract_feature_vector(augmented_base_signup_np_image, model_name="Facenet512", face_detector_backend="centerface") for augmented_base_signup_np_image in augmented_base_signup_np_images]
        base_login_np_emb = dnn.extract_feature_vector(base_login_np_image, model_name="Facenet512", face_detector_backend="centerface")
        false_login_np_emb = dnn.extract_feature_vector(false_login_np_image, model_name="Facenet512", face_detector_backend="centerface")

        return (
            base_signup_np_embs,
            base_login_np_emb,
            false_login_np_emb,
        )



    def load_one_sample(self, people_struct_pair):
        (
            (base_images, base_length),
            (false_images, false_length),
        ) = self._split_true_and_false_embeddings(people_struct_pair)

        (
            base_signup_np_image,
            base_login_np_image,
            false_login_np_image,
        ) = self._divide_into_register_login_samples(
            base_images,
            base_length,
            false_images,
            false_length,
        )

        return self._do_image_augmentation(base_signup_np_image, base_login_np_image, false_login_np_image)
