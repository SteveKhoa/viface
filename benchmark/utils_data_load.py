import abc

class DataLoad:
    @abc.abstractmethod
    def load_one_sample(self, people_struct_pair) -> list:
        pass