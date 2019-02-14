"""
This module includes HMM class which defines
the desired Hidden Markov Model
"""

import matplotlib.pyplot as plt
from hmmlearn import hmm
from utils import *


class HMM:

    def __init__(self, n_state, ndim):

        """

        :param n_state: Integer, number of hidden states
        :param ndim: Integer, dimension of the states and observations

        """

        self.n_state = n_state
        self.ndim = ndim
        self.prior_probs = random_prior_prob_generation(self.n_state)
        self.transition_probs = random_transition_prob_generation(self.n_state)
        self.means = mean_vector_generation(self.n_state, self.ndim)
        self.covars = cov_matrix_generation(self.n_state, self.ndim)

    def sample(self, size_sample):

        """

        :param size_sample: Integer, number of elements in the sample
        :return: 2D Array
        """

        model = hmm.GaussianHMM(n_components=self.n_state, covariance_type="full")

        model.startprob_ = self.prior_probs
        model.transmat_ = self.transition_probs
        model.means_ = self.means
        model.covars_ = self.covars

        x = model.sample(size_sample)[0]

        return x

    def plot_sample(self, x):

        """

        :param x: 2D Array sampled data
        :return: Plot
        """

        # Plot the sampled data
        fig = plt.figure(figsize=plt.figaspect(0.5))

        ax = fig.add_subplot(1, 2, 1, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=7, c='green')
        for i in range(self.n_state):
            ax.scatter(self.means[i, 0], self.means[i, 1], self.means[i, 2], s=50)

        ax = fig.add_subplot(1, 2, 2, projection='3d')
        ax.scatter(x[:, 3], x[:, 4], x[:, 5], s=7, c='green')
        for i in range(self.n_state):
            ax.scatter(self.means[i, 3], self.means[i, 4], self.means[i, 5], s=50)

        plt.show()
