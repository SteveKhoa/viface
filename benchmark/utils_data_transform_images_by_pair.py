import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], ".."))

from collections import defaultdict
import random
import re
import sklearn
from benchmark import utils_time_measurement
from benchmark.constant import REMOVE_IDENTITIES_LESS_THAN_N_SAMPLES


class DataTransformImagesByPair:
    def __init__(self, n_pairs_to_load: int, file_dictionary: dict):
        super().__init__()

        self.n_pairs = n_pairs_to_load
        self.file_dictionary = file_dictionary

    def _remove_folders_less_than_n_samples(self, file_dictionary: dict):
        """This utility function group the file paths into distinct values (people names)
        and filter out people with only one sample (because one sample is not useful
        for downstream tasks).
        """
        keys = list(file_dictionary.keys())
        s = defaultdict(int)
        for key in keys:
            justname = "_".join(key.split("/")[-1].split(".")[0].split("_")[:-1])
            s[justname] += 1

        filtered_keys = dict(list(filter(lambda x: x[1] > REMOVE_IDENTITIES_LESS_THAN_N_SAMPLES, s.items()))).keys()

        print("DataTransformImagesByPair: number of people after filtered=", len(filtered_keys))
        return filtered_keys

    def _select_random_people_pair(self, filepaths, name_list):
        sampled = random.sample(name_list, 2)

        # Append prefix and postfix with wildcard so that regex can match it
        sampled = [".*" + sample + ".*" for sample in sampled]

        # Preselect so that we don't have to run filter multiple times
        selected = list(
            filter(
                lambda fpath: re.match(sampled[0], fpath)
                or re.match(sampled[1], fpath),
                filepaths,
            )
        )

        return selected, sampled

    @utils_time_measurement.decorator_time_measurement_log("DataTransformImagesByPair: transform the datasets")
    def transform(self):
        names = self._remove_folders_less_than_n_samples(
            self.file_dictionary
        )
        filepaths = list(self.file_dictionary.keys())

        people_struct_pair_list = []
        for i in range(self.n_pairs):
            if i % 100 == 0:
                print(f"DataTransformImagesByPair: {i}\{self.n_pairs}")

            selected, sampled = self._select_random_people_pair(filepaths, names)

            people_by_name = dict()
            for name in sampled:
                images = [
                    self.file_dictionary[fname]
                    for fname in filter(
                        lambda fpath: re.match(name, fpath),
                        selected,
                    )
                ]

                people_by_name[name] = {
                    "images": images,
                    "n_images": len(images),
                }

            # Record this people
            people_struct_pair = list(people_by_name.items())
            people_struct_pair_list += [people_struct_pair]

        return people_struct_pair_list
