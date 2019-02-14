from numpy.core.multiarray import ndarray
import sklearn.datasets
import numpy as np


def sum_normalize(arr):

    """
    This function normalizes the given 1D array so that 
    the sum of the elements in it is equal to 1. Instead
    of inplacing, it returns a new 1D array.

    :param arr: 1D Numpy Array
    :return: 1D Numpy Normalized Array
    """

    return np.array([i/np.sum(arr) for i in arr])


def random_prior_prob_generation(nstate):

    """
    This function returns a prior distribution based on the
    given number of states such that sum of the all probabilities
    is equal to 1.
    
    :param nstate: Integer
    :return: 1D Numpy Normalized Array
    """

    return sum_normalize(np.random.uniform(0, 1, nstate))


def random_transition_prob_generation(nstate):

    """

    :param nstate: Integer
    :return: 2D Numpy Array
    """

    trans_prob = []
    for i in range(nstate):
        trans_prob.append(sum_normalize(np.random.uniform(0, 1, nstate)))

    return np.array(trans_prob)


def mean_vector_generation(nstate, ndim):

    """

    :param nstate: Integer
    :param ndim: Integer
    :return: 2D Numpy Array
    """

    means = []
    for i in range(nstate):
        means.append(np.random.uniform(-10, +10, ndim))

    return np.array(means)


def cov_matrix_generation(nstate, ndim):

    """

    :param nstate: Integer
    :param ndim: Integer
    :return: 3D Numpy Array
    """

    covs: ndarray = np.empty(shape=(nstate, ndim, ndim))
    for i in range(nstate):
        covs[i, :] = sklearn.datasets.make_spd_matrix(ndim)

    return covs
