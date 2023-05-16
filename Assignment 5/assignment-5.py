def voting_ensemble(classifiers):
    """
        Takes as a parameter a list of classifiers, and 
        returns a new classifier that reports, for a given 
        nput, the winning vote amongst all the given 
        classifiers on that input
    """
    def new_classifier(input):

        results = [classifier(input) for classifier in classifiers]
        max_occurrences = max(results.count(x) for x in results)
        max_results = [x for x in results if results.count(x) == max_occurrences]

        return min(max_results)

    return new_classifier

import hashlib
import numpy as np
def pseudo_random(seed=0xDEADBEEF):
    """Generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed)/0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits)/0xffffffff
        yield r

def bootstrap(dataset, sample_size):
    """
        Samples the given dataset to produce samples of sample_size
    """
    number_generator = pseudo_random()
    while True:
        sample = []
        
        while len(sample) < sample_size:
            next_number = next(number_generator)
            index = int(next_number * len(dataset))
            sample.append(dataset[index])
        yield np.array(sample)

def bagging_model(learner, dataset, n_models, sample_size):
    """
        Returns a new model based on the learner, but replicated 
        n_models times over the bootstrapped dataset of size sample_size.
    """

    models = []

    b_strap = bootstrap(dataset, sample_size)
    for i in range(n_models):
        sample = next(b_strap)
        models.append(learner(sample))
    
    return voting_ensemble(models)

class weighted_bootstrap:
    def __init__(self, dataset, weights, sample_size):
        self.dataset = dataset
        self.weights = weights
        self.sample_size = sample_size
        self.number_generator = pseudo_random()

    def __iter__(self):
        return self
    
    def __next__(self):
        sample = []
        indexes = []
        weight_running_sum = [sum(self.weights[:i + 1]) for i in range(len(self.weights))]
        
        while len(sample) < self.sample_size:
            r = (next(self.number_generator) * sum(self.weights))
            for index in range(len(weight_running_sum)):
                if weight_running_sum[index] > r:
                    break
            indexes.append(index)
            sample.append(self.dataset[index])
        return np.array(sample), indexes

import math
def adaboost(learner, dataset, n_models):
    """
        Builds a boosted ensemble model consisting of n_models simple 
        models made from the learner, which were trained on weighted 
        bootstrapped samples from the dataset.
    """
    models = []

    bootstrapper = weighted_bootstrap(dataset, [1/len(dataset) for _ in range(len(dataset))], len(dataset))

    for i in range(n_models):
        sample, indexes = next(bootstrapper)
        new_model = learner(sample)
        error = 0
        for i in indexes:
            data = dataset[i]
            if new_model(data[:-1]) != data[-1]:
                error += bootstrapper.weights[i]

        models.append((new_model, error))
        if error == 0 or error >= 0.5:
            break
        for i in range(len(dataset)):
            instance = dataset[i]
            if new_model(instance[:-1]) == instance[-1]:
                bootstrapper.weights[i] *= (error / (1-error))

        bootstrapper.weights = [weight / sum(bootstrapper.weights) for weight in bootstrapper.weights]
    
    def boosted_ensemble_model(input):
        weights = {}
        for data in dataset:
            weights[data[-1]] = 0

        for model, error in models:
            output = model(input)
            if error == 0:
                weights[output] += math.inf
            else:
                weights[output] += (-1 * math.log(error / (1-error)))
        return max(weights.keys(), key=lambda x: weights[x])

    return boosted_ensemble_model






    


if __name__ == "__main__":
    	
    import sklearn.datasets
    import sklearn.utils
    import sklearn.linear_model

    digits = sklearn.datasets.load_digits()
    data, target = sklearn.utils.shuffle(digits.data, digits.target, random_state=3)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]
    dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

    def linear_learner(dataset):
        features, target = dataset[:, :-1], dataset[:, -1]
        model = sklearn.linear_model.SGDClassifier(random_state=1, max_iter=1000, tol=0.001).fit(features, target)
        return lambda v: model.predict(np.array([v]))[0]

    boosted = adaboost(linear_learner, dataset, 10)
    output, expected = [], []
    for (v, c) in zip(test_data, test_target):
        output.append(int(boosted(v)))
        expected.append(c)
        print(int(boosted(v)), c)
    print()

    if not output == expected:
        print("Test 1 failed")
        [print(output[x], expected[x]) for x in range(len(output))]
        print()

    import sklearn.datasets
    import sklearn.utils
    import sklearn.linear_model
    iris = sklearn.datasets.load_iris()
    data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]
    dataset = np.hstack((train_data, train_target.reshape((-1, 1))))
    def linear_learner(dataset):
        features, target = dataset[:, :-1], dataset[:, -1]
        model = sklearn.linear_model.SGDClassifier(random_state=1, max_iter=1000, tol=0.001).fit(features, target)
        return lambda v: model.predict(np.array([v]))[0]
    boosted = adaboost(linear_learner, dataset, 10)
    output, expected = [], []
    for (v, c) in zip(test_data, test_target):
        output.append(int(boosted(v)))
        expected.append(c)
        print(int(boosted(v)), c)
    print()
    if not output == expected:
        print("Test 2 failed")
        [print(output[x], expected[x]) for x in range(len(output))]
        print()

    	
    import sklearn.datasets
    import sklearn.utils
    import sklearn.tree

    wine = sklearn.datasets.load_wine()
    data, target = sklearn.utils.shuffle(wine.data, wine.target, random_state=3)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]
    dataset = np.hstack((train_data, train_target.reshape((-1, 1))))

    def tree_learner(dataset):
        features, target = dataset[:, :-1], dataset[:, -1]
        model = sklearn.tree.DecisionTreeClassifier(random_state=1).fit(features, target)
        return lambda v: model.predict(np.array([v]))[0]

    boosted = adaboost(tree_learner, dataset, 10)
    output, expected = [], []
    for (v, c) in zip(test_data, test_target):
        output.append(int(boosted(v)))
        expected.append(c)
        print(int(boosted(v)), c)
    print()
    if not output == expected:
        print("Test 3 failed")
        [print(output[x], expected[x]) for x in range(len(output))]
            

