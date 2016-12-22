"""
Neural gas algorithm to adapt markers to data sampled from a probability
distribution P(x)
"""
import numpy as np


def adapt_NG(neurons, sampled_data, epsilon, plambda, data_cicles):
    """
    For each data x in sampled_data, the nodes in
    neurons are adapted according to:
    w_{ik}^{t+1} = w_{ik}^t + \epsilon e^{-k/\lambda}(x - w_{ik}^t)

    :param neurons: numpy array of markers w
    :param sampled_data: 2D numpy array of points sampled from P(x)
    :param epsilon: adaptation step size
    :param plambda: neighborhood range
    :param data_cicles: times the adaptation algorithm will run through all
                        sampled data.
    :return: None.  Neurons are modified in place.
    """
    num_points = len(sampled_data)
    num_neurons = len(neurons)
    proportions = epsilon * np.exp(-np.arange(num_neurons)/plambda)
    proportions = proportions[..., np.newaxis]
    for cycle in range(data_cicles):
        rand_indices = np.arange(0, num_points)
        np.random.shuffle(rand_indices)
        for x in sampled_data[rand_indices]:
            distances = np.linalg.norm(neurons - x, axis=1)
            sorted_indices = np.argsort(distances)
            sorted_view = neurons[sorted_indices]
            sorted_view = sorted_view + proportions * (x - sorted_view)
            neurons[sorted_indices] = sorted_view
