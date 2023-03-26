class DTNode:
    def __init__(self, decision):
        self.decision = decision
        self.children = []
    
    def predict(self, input): return self.children[self.decision(input)].predict(input) if callable(self.decision) else self.decision
        
    def leaves(self): return 1 if self.children == [] else sum(c.leaves() for c in self.children)
             


def partition_by_feature_value(dataset, feature_index):
    partitions = dict()
    for pair in dataset:
        if (pair[0][feature_index]) in partitions.keys():
            partitions[pair[0][feature_index]].append(pair)
        else:
            partitions[pair[0][feature_index]] = [pair]

    keys = list(partitions.keys())
    values = list(partitions.values())
    func = lambda x : keys.index(x[feature_index]) 
    return func, values

def classification_proportion(dataset, classification):
    k = len([x for x in dataset if x[1] == classification])
    return k / len(dataset)

def all_classification_proportions(dataset):
    output = dict()
    for classification in [x[1] for x in dataset]:
        if classification not in output.keys():
            output[classification] = classification_proportion(dataset, classification)
    return output

def misclassification(dataset):
    return 1 - max(all_classification_proportions(dataset).values())

def gini(dataset):
    return sum([pk * (1-pk) for pk in all_classification_proportions(dataset).values()])

from math import log2
def entropy(dataset):
    return -1 * sum([pk * log2(pk) for pk in all_classification_proportions(dataset).values()])

from statistics import mode
def train_tree(dataset, criterion, features=None):
    if features is None:
        features = [x for x in range(len(dataset[0][0]))]
    classes = [x[1] for x in dataset]
    if len(set(classes)) == 1:
        return DTNode(classes[0])
    elif len(dataset) == 0:
        return DTNode(mode(classes))
    else:
        min_error = None
        best_separator = None
        best_partition = None
        best_feature = None
        for feature in features:
            error = 0
            separator, partitions = partition_by_feature_value(dataset, feature)
            for partition in partitions:
                error += (len(partition) / len(dataset)) * criterion(partition)
            
            if min_error is None or min_error > error:
                min_error = error
                best_separator = separator
                best_partition = partitions
                best_feature = feature
        R = DTNode(best_separator)
        features = features.copy()
        features.remove(best_feature)

        for partition in best_partition:
            R.children.append(train_tree(partition, criterion, features))
        return R






if __name__ == "__main__":
    # The following (leaf) node will always predict True
    node = DTNode(True) 

    # Prediction for the input (1, 2, 3):
    x = (1, 2, 3)
    assert node.predict(x) == True

    # Sine it's a leaf node, the input can be anything. It's simply ignored.
    assert node.predict(None) == True

    yes_node = DTNode("Yes")
    no_node = DTNode("No")
    tree_root = DTNode(lambda x: 0 if x[2] < 4 else 1)
    tree_root.children = [yes_node, no_node]

    assert tree_root.predict((False, 'Red', 3.5)) == "Yes"
    assert tree_root.predict((False, 'Green', 6.1)) == "No"

    n = DTNode(True)
    assert n.leaves() == 1

    t = DTNode(True)
    f = DTNode(False)
    n = DTNode(lambda v: 0 if not v else 1)
    n.children = [t, f]
    assert n.leaves() == 2

    dataset = [
    ((True, True), False),
    ((True, False), True),
    ((False, True), True),
    ((False, False), False),
    ]
    f, p = partition_by_feature_value(dataset,  0)
    assert sorted(sorted(partition) for partition in p) == [[((False, False), False), ((False, True), True)], [((True, False), True), ((True, True), False)]]

    partition_index = f((True, True))
    # Everything in the "True" partition for feature 0 is true
    assert (all(x[0]==True for x,c in p[partition_index]))
    partition_index = f((False, True))
    # Everything in the "False" partition for feature 0 is false
    assert (all(x[0]==False for x,c in p[partition_index]))

    dataset = [
    (("a", "x", 2), False),
    (("b", "x", 2), False),
    (("a", "y", 5), True),
    ]
    f, p = partition_by_feature_value(dataset, 1)
    assert sorted(sorted(partition) for partition in p) == [[(('a', 'x', 2), False), (('b', 'x', 2), False)], [(('a', 'y', 5), True)]]
    partition_index = f(("a", "y", 5))
    # everything in the "y" partition for feature 1 has a y
    assert (all(x[1]=="y" for x, c in p[partition_index]))

    data = [
        ((False, False), False),
        ((False, True), True),
        ((True, False), True),
        ((True, True), False)
    ]
    assert ("{:.4f}".format(misclassification(data))) == "0.5000"
    assert ("{:.4f}".format(gini(data))) == "0.5000"
    assert ("{:.4f}".format(entropy(data))) == "1.0000"

    dataset = [
    ((True, True), False),
    ((True, False), True),
    ((False, True), True),
    ((False, False), False)
    ]
    # t = train_tree(dataset, misclassification)
    # print(t.predict((True, False)))
    # print(t.predict((False, False)))

    training_examples = [
        (("Sunny", "Hot", "High", "Weak"), "No"),
        (("Sunny", "Hot", "High", "Strong"), "No"),
        (("Overcast", "Hot", "High", "Weak"), "Yes"),
        (("Rain", "Mild", "High", "Weak"), "Yes"),
        (("Rain", "Cool", "Normal", "Weak"), "Yes"),
        (("Rain", "Cool", "Normal", "Strong"), "No"),
        (("Overcast", "Cool", "Normal", "Strong"), "Yes"),
        (("Sunny", "Mild", "High", "Weak"), "No"),
        (("Sunny", "Cool", "Normal", "Weak"), "Yes"),
        (("Rain", "Mild", "Normal", "Weak"), "Yes"),
        (("Sunny", "Mild", "Normal", "Strong"), "Yes"),
        (("Overcast", "Mild", "High", "Strong"), "Yes"),
        (("Overcast", "Hot", "Normal", "Weak"), "Yes"),
        (("Rain", "Mild", "High", "Strong"), "No")
    ]

    t = train_tree(training_examples, misclassification)
    for example in training_examples:
        assert (t.predict(example[0])) == example[1]


    examples = [
        ((True, True, True), True),
        ((True, True, False), False),
        ((True, False, True), True),
        ((True, False, False), False),
        ((False, True, True), True),
        ((False, True, False), False),
        ((False, False, True), True),
        ((False, False, False), False)
    ]

    g = train_tree(examples, misclassification)
    for example in examples:
        assert (g.predict(example[0])) == example[1]

