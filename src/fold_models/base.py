from copy import deepcopy

from fold.models.base import Model


class Wrapper(Model):
    def clone(self) -> "Wrapper":
        return deepcopy(self)
