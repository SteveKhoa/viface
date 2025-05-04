import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from random import randint


class DataLoadEmbeddingsByPair:
    def _split_true_and_false_embeddings(self, people_struct_pair):
        # Get a true identity and a false identity
        true_identity = 0
        false_identity = 1

        # Acquire base
        base_embeddings = people_struct_pair[true_identity][-1]["embeddings"]
        base_length = people_struct_pair[true_identity][-1]["n_embeddings"]

        # Acquire false
        false_embeddings = people_struct_pair[false_identity][-1]["embeddings"]
        false_length = people_struct_pair[false_identity][-1]["n_embeddings"]

        return (
            (base_embeddings, base_length),
            (false_embeddings, false_length),
        )

    def _divide_into_register_login_samples(
        self,
        base_embeddings,
        base_length,
        false_embeddings,
        false_length,
    ):
        # Select two random samples in reference (base) identity
        base_select_signup = randint(0, base_length - 1)
        base_select_login = randint(0, base_length - 1)
        while (
            base_select_login == base_select_signup
        ):  # make sure two samples are not the same
            base_select_login = randint(0, base_length - 1)

        base_signup_np_emb = base_embeddings[base_select_signup]
        base_login_np_emb = base_embeddings[base_select_login]

        # Select one random sample from false identity
        false_select_login = randint(0, false_length - 1)
        false_login_np_emb = false_embeddings[false_select_login]

        return (
            base_signup_np_emb,
            base_login_np_emb,
            false_login_np_emb,
        )

    def load_one_sample(self, people_struct_pair):
        (
            (base_embeddings, base_length),
            (false_embeddings, false_length),
        ) = self._split_true_and_false_embeddings(people_struct_pair)

        return self._divide_into_register_login_samples(
            base_embeddings,
            base_length,
            false_embeddings,
            false_length,
        )
