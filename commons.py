import os
import re
import numpy as np
from typing import List
import bisect


class Folder:
    """
    Simple folder management, as a linear list of files (sorted by file names).
    """

    def __init__(self, folderpath: str = "./lfw.embeddings"):
        self.orgpath = folderpath
        self.files = np.array(
            list(sorted(filter(lambda x: x != ".DS_Store", os.listdir(folderpath))))
        )

    def nitems(self):
        return len(self.files)

    def query(self, filepattern: str):
        """Returns metadata of the query and cache the query. The next `commit()` will
        actually load the numpy files.

        Pass in `.*` to query all the files
        """
        self.selected = list(
            filter(
                lambda x: re.match(
                    f"{filepattern}",
                    x,
                ),
                self.files,
            )
        )
        return self.selected

    def query_byidx(self, indexes: List[int]):
        """
        Same as query, but accessed based on list indexes.
        """
        self.selected = []
        self.selected = self.files[indexes]
        return list(self.selected)

    def commit(self, cache=False) -> np.ndarray:
        """
        Load numpy files of the latest query into the list.
        """
        values = [np.load(os.path.join(self.orgpath, select)) for select in self.selected]
        if cache == True:
            self.commit_cache = values
        else:
            pass
        return values


if __name__ == "__main__":
    pass
