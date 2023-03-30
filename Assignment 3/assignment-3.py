def dot_product(input1, input2):
    """
        Performs the dot product on two arrays
    """

    return sum([input1[i] * input2[i] for i in range(len(input1))])

def linear_regression_1d(input):
    """
        Takes in a list of pairs, where the first value in each pair is the 
        feature value and the second is the response value. Return a pair 
        (m, c) where m is the slope of the line of least squares fit, and 
        c is the intercept of the line of least squares fit. 
    """
    n = len(input)

    x, y = zip(*input)

    m = (n * dot_product(x, y) - (sum(x) * sum(y))) / (n * dot_product(x, x) - (sum(x) ** 2))
    c = (sum(y) - m * sum(x)) / n
    return m, c


import numpy as np

def linear_regression(xs, ys):
    """
        Takes two numpy arrays as input: the first is the input part of the 
        training data which is an min array (design matrix), while the second 
        is the output part of the training data which is a one-dimensional 
        array (vector) with m elements. Return the one-dimensional array 
        (vector) θ, with (n + 1) elements, which contains the least-squares 
        regression coefficients of the features; the first ("extra") value is the intercept.
    """
    X = np.c_[np.ones(len(ys)), xs]

    return np.linalg.inv(X.T @ X) @ X.T @ ys

import math
def logistic_regression(xs, ys, alpha, num_interations):

    sigmoid = lambda x: 1 / (1 + math.exp(-x))




xs = np.arange(5).reshape((-1, 1))
ys = np.arange(1, 11, 2)
print(linear_regression(xs, ys))


xs = np.array([[1, 2, 3, 4],
               [6, 2, 9, 1]]).T
ys = np.array([7, 5, 14, 8]).T
print(linear_regression(xs, ys))