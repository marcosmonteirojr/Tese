from deslib.des.base import DES
import numpy as np

class ALL(DES):
    def __init__(self, pool_classifiers):
        super(ALL, self).__init__(pool_classifiers)
        self.name = 'ALL'
        self.indices = list(range(self.n_classifiers))
        self.competences = np.ones(self.n_classifiers)

    def estimate_competence(self, query):
        return self.competences

    def select(self, competences):
        return self.indices
