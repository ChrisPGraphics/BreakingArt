import copy
import pickle
import typing


class SavableObject:
    @classmethod
    def load(cls, filename: str) -> 'typing.Self':
        with open(filename, 'rb') as f:
            return pickle.loads(f.read())

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            f.write(pickle.dumps(self))

    def copy(self, deep_copy: bool = True) -> 'typing.Self':
        if deep_copy:
            return copy.deepcopy(self)

        else:
            return copy.copy(self)
