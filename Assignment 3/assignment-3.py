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

def linear_regression(xs, ys, basis_functions=[lambda x: x], penalty=0):
    """
        takes xs (training input), ys (training output), and basis_functions 
        which is a list of basis functions and returns a one-dimensional array 
        (vector) of coefficients where the first elements is the offset and 
        the rest are the coefficients of the corresponding basis functions. 
        Each basis function takes a complete input vector and returns a scalar 
        which is the value of the basis function for that input. When functions 
        are not provided, the algorithm should behave as an ordinary linear 
        regression using normal equations.
    """
    
    new_xs = [np.array([]) for _ in range(len(xs))]
    
    for func in basis_functions:
        for i in range(len(xs)):
            new_xs[i] = np.append(new_xs[i], [func(xs[i])])

    X = np.c_[np.ones(len(ys)), new_xs]

    return np.linalg.inv(X.T @ X + penalty * np.identity(len(X[0]))) @ X.T @ ys

import math
def logistic_regression(xs, ys, alpha, num_iterations):
    """
        Takes as input a training data set and returns a model that we can 
        use to classify new feature vectors. The xs and ys parameters are 
        the same as in previous questions: a two-dimensional array and a 
        one-dimensional array. The alpha parameter is the training/learning 
        rate, while the num_iterations is the number of iterations to 
        perform - that is, how many times to loop over the entire dataset.
    """
    sigmoid = lambda x: 1 / (1 + math.exp(-x))

    xs = np.c_[np.ones(len(ys)), xs]
    theta = np.zeros(len(xs[0]))
    for _ in range(num_iterations):
        for i in range(len(xs)):
            theta += alpha * (ys[i] - sigmoid(theta @ xs[i])) * xs[i]
    return lambda x: sigmoid(theta @ np.insert(x, 0, 1))
    

    






if __name__ == "__main__":
    # we set the seed to some number so we can replicate the computation
    np.random.seed(0)

    xs = np.arange(-1, 1, 0.1).reshape(-1, 1)
    m, n = xs.shape
    # Some true function plus some noise:
    ys = (xs**2 - 3*xs + 2 + np.random.normal(0, 0.5, (m, 1))).ravel()

    functions = [lambda x: x[0], lambda x: x[0]**2, lambda x: x[0]**3, lambda x: x[0]**4,
        lambda x: x[0]**5, lambda x: x[0]**6, lambda x: x[0]**7, lambda x: x[0]**8]

    for penalty in [0, 0.01, 0.1, 1, 10]:
        with np.printoptions(precision=5, suppress=True):
            print(linear_regression(xs, ys, basis_functions=functions, penalty=penalty)
                .reshape((-1, 1)), end="\n\n")
