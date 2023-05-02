from collections import namedtuple
import numpy as np
from statistics import mean
import hashlib

class ConfusionMatrix(namedtuple("ConfusionMatrix",
                                 "true_positive false_negative "
                                 "false_positive true_negative")):

    def __str__(self):
        elements = [self.true_positive, self.false_negative,
                   self.false_positive, self.true_negative]
        return ("{:>{width}} " * 2 + "\n" + "{:>{width}} " * 2).format(
                    *elements, width=max(len(str(e)) for e in elements))
    
def confusion_matrix(classifier, dataset):
    """
        Takes a classifier and a test dataset, and returns a confusion
        matrix capturing how well the classifier classified the dataset. 
        The confusion matrix object must be created by calling 
        ConfusionMatrix(tp, fn, fp, tn).
    """
    tp, fn, fp, tn = 0, 0, 0, 0

    for sample, classification in dataset:
        classifier_result = classifier(sample)
        if classifier_result == 0:
            if classification == 0:
                tn += 1
            else:
                fn += 1
        else:
            if classification == 1:
                tp += 1
            else:
                fp += 1
    
    return ConfusionMatrix(tp, fn, fp, tn)

def roc_non_dominated(classifiers):
    """
        Takes a collection of classifiers and returns only those 
        classifiers that are not dominated by any other classifier 
        in the collection. A classifier is represented as a pair 
        (classifier_name, confusion_matrix), where classifier_name 
        is a string, and confusion_matrix is a named tuple representing 
        the two-by-two classification confusion matrix.
    """
    return [classifiers[i] for i in range(len(classifiers)) if not any(classifiers[i][1].true_positive < classifiers[j][1].true_positive and classifiers[i][1].false_positive > classifiers[j][1].false_positive for j in (list(range(i)) + list(range(i+1, len(classifiers)))))]

def k_means(dataset, centroids):
    """
        Takes a dataset and k centroids and returns the tuple of updated centroids
    """

    old_centroids = [[] for _ in centroids]
    new_centroids = centroids

    while not np.array_equal(new_centroids, old_centroids):
        classes = [[] for _ in new_centroids]

        for point in dataset:
            nearest_centroid_index = min(range(len(new_centroids)), key=lambda x: np.linalg.norm(point - new_centroids[x])) 
            classes[nearest_centroid_index].append(point)
        
        old_centroids = new_centroids
        new_centroids = []

        
        new_centroids = [np.array([mean(x[j] for x in classes[i]) for j in range(len(old_centroids[i]))]) if classes[i] else old_centroids[i] for i in range(len(old_centroids))]
    return old_centroids

def goodness(clusters):
    
    compactness = []

    for cluster in clusters:
        max_distance = 0
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                distance = np.linalg.norm(cluster[i] - cluster[j])
                if distance > max_distance:
                    max_distance = distance
        compactness.append(max_distance)


    separation = mean(min(x) for x in distances)

    compactness = mean(max(x) for x in distances)

    return separation / compactness

def cluster_points(centroids, dataset):
    classes = [[] for _ in centroids]

    for point in dataset:
        nearest_centroid_index = min(range(len(centroids)), key=lambda x: np.linalg.norm(point - centroids[x])) 
        classes[nearest_centroid_index].append(point)
    
    return classes

def pseudo_random(seed=0xdeadbeef):
    """generate an infinite stream of pseudo-random numbers"""
    state = (0xffffffff & seed)/0xffffffff
    while True:
        h = hashlib.sha256()
        h.update(bytes(str(state), encoding='utf8'))
        bits = int.from_bytes(h.digest()[-8:], 'big')
        state = bits >> 32
        r = (0xffffffff & bits)/0xffffffff
        yield r

def generate_random_vector(bounds, r):
    return np.array([(high - low) * next(r) + low for low, high in bounds])


def k_means_random_restart(dataset, k, restarts, seed=None):
    bounds = list(zip(np.min(dataset, axis=0), np.max(dataset, axis=0)))
    r = pseudo_random(seed=seed) if seed else pseudo_random()
    models = []
    for _ in range(restarts):
        random_centroids = tuple(generate_random_vector(bounds, r) 
                                 for _ in range(k))
        new_centroids = k_means(dataset, random_centroids)
        clusters = cluster_points(new_centroids, dataset)
        if any(len(c) == 0 for c in clusters):
            continue
        models.append((goodness(clusters), new_centroids))
    return max(models, key=lambda x: x[0])[1]
        
        




if __name__ == "__main__":
    
    # dataset = [
    #     ((0.8, 0.2), 1),
    #     ((0.4, 0.3), 1),
    #     ((0.1, 0.35), 0),
    # ]
    # print(confusion_matrix(lambda x: 1, dataset))
    # print()
    # print(confusion_matrix(lambda x: 1 if x[0] + x[1] > 0.5 else 0, dataset))

    # classifiers = [
    #     ("Red", ConfusionMatrix(60, 40, 
    #                             20, 80)),
    #     ("Green", ConfusionMatrix(40, 60, 
    #                             30, 70)),
    #     ("Blue", ConfusionMatrix(80, 20, 
    #                             50, 50)),
    # ]
    # print(sorted(label for (label, _) in roc_non_dominated(classifiers)))

    # classifiers = []
    # with open("roc_small.data") as f:
    #     for line in f.readlines():
    #         label, tp, fn, fp, tn = line.strip().split(",")
    #         classifiers.append((label,
    #                             ConfusionMatrix(int(tp), int(fn),
    #                                             int(fp), int(tn))))
    # print(sorted(label for (label, _) in roc_non_dominated(classifiers)))

    # classifiers = []
    # with open("roc.data") as f:
    #     for line in f.readlines():
    #         label, tp, fn, fp, tn = line.strip().split(",")
    #         classifiers.append((label,
    #                             ConfusionMatrix(int(tp), int(fn),
    #                                             int(fp), int(tn))))
    # print(sorted(label for (label, _) in roc_non_dominated(classifiers)))

    # dataset = np.array([
    #     [0.1, 0.1],
    #     [0.2, 0.2],
    #     [0.8, 0.8],
    #     [0.9, 0.9]
    # ])
    # centroids = (np.array([0., 0.]), np.array([1., 1.]))
    # for c in k_means(dataset, centroids):
    #     print(c)

    # dataset = np.array([
    #     [0.125, 0.125],
    #     [0.25, 0.25],
    #     [0.75, 0.75],
    #     [0.875, 0.875]
    # ])
    # centroids = (np.array([0., 1.]), np.array([1., 0.]))
    # for c in k_means(dataset, centroids):
    #     print(c)

    # dataset = np.array([
    #     [0.1, 0.3],
    #     [0.4, 0.6],
    #     [0.1, 0.2],
    #     [0.2, 0.1]
    # ])
    # centroids = (np.array([2., 5.]),)
    # for c in k_means(dataset, centroids):
    #     print(c)

    # import sklearn.datasets
    # import sklearn.utils

    # iris = sklearn.datasets.load_iris()
    # data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    # train_data, train_target = data[:-5, :], target[:-5]
    # test_data, test_target = data[-5:, :], target[-5:]


    # centroids = (
    #     np.array([5.8, 2.5, 4.5, 1.5]),
    #     np.array([6.8, 3.0, 5.7, 2.1]),
    #     np.array([5.0, 3.5, 1.5, 0.5])
    # )
    # for c in k_means(train_data, centroids):
    #     print(c)

    dataset = np.array([
        [0.1, 0.1],
        [0.2, 0.2],
        [0.8, 0.8],
        [0.9, 0.9]
    ])
    centroids = k_means_random_restart(dataset, k=2, restarts=5)

    for c in sorted([f"{x:8.3}" for x in centroid] for centroid in centroids):
        print(" ".join(c))

    import sklearn.datasets
    import sklearn.utils

    iris = sklearn.datasets.load_iris()
    data, target = sklearn.utils.shuffle(iris.data, iris.target, random_state=0)
    train_data, train_target = data[:-5, :], target[:-5]
    test_data, test_target = data[-5:, :], target[-5:]

    centroids = k_means_random_restart(train_data, k=3, restarts=10)


    # We suggest you check which centroid each 
    # element in test_data is closest to, then see test_target.
    # Note cluster 0 -> label 1
    #      cluster 1 -> label 2
    #      cluster 2 -> label 0

    for c in sorted([f"{x:7.2}" for x in centroid] for centroid in centroids):
        print(" ".join(c))