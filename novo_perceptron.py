# coding=utf-8

# Author: Marcelo T. Pereira <marcelo.trier@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.linear_model import Perceptron
from scipy.special import expit

'''
    Aqui na versão 4 estou reimplementando o Perceptron para aceitar uma função predict_proba,
    que sem ela NAADA FUNCIONAAA!!
'''

# esta é uma versão do Perceptron ao invés de utilizar CalibratedClassifierCV, como abaixo:
# model = CalibratedClassifierCV(Perceptron(), cv=3, method='sigmoid')

# PEREIRA PERCEPTRON: PPerceptron
class PPerceptron(Perceptron):

    def predict_proba(self, x_data):
        """Probability estimation for OvR logistic regression.

        Positive class probabilities are computed as
        1. / (1. + np.exp(-self.decision_function(X)));
        multiclass is handled by normalizing that over all classes.
        """
        # a função abaixo significa: predict_proba_logistic_regression!
        return self._predict_proba_lr(x_data)

    # a função abaixo é o mesmo que predict_proba acima, só que feita passo-a-passo
    def predict_proba_PEREIRA(self, x_data):
        # EXPIT FUNCTION
        # The expit function, also known as the logistic function, is defined as
        # expit(x) = 1/(1+exp(-x)). It is the inverse of the logit function.

        values = self.decision_function(x_data)
        probs = np.round(expit(values), 4)
        ss = np.sum(probs, axis=1) # soma as probabilidades de cada classe
        ss = np.atleast_2d(ss).T   # agora pega essa soma, faz transpose...
        probs /= ss                # e divide! só isso! ;P
        return probs

