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


data = [(1, 4), (2, 7), (3, 10)]
m, c = linear_regression_1d(data)
print(m, c)
print(4 * m + c)