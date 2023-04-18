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
    keys, values = zip(*partitions.items())
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

# def train_tree(dataset, criterion, features=None):
#     if features is None: features = [x for x in range(len(dataset[0][0]))]

#     if len(set([x[1] for x in dataset])) == 1:
#         return DTNode([x[1] for x in dataset][0])
#     elif len(dataset) == 0:
#         return DTNode(__import__("statistics").mode([x[1] for x in dataset]))
#     else:
#         min_error, best_separator, best_partition, best_feature = None, None, None, None
#         for feature in features:
#             separator, partitions = partition_by_feature_value(dataset, feature)
#             error = sum((len(partition) / len(dataset)) * criterion(partition) for partition in partitions)
#             if min_error is None or min_error > error: min_error, best_separator, best_partition, best_feature = error, separator, partitions, feature
#         features = features.copy()
#         features.remove(best_feature)
#         R = DTNode(best_separator)
        
#         [R.children.append(train_tree(partition, criterion, features)) for partition in best_partition]
#         return R

def train_tree(d,c,f=None):
 f=f or [*range(len(d[0][0]))]
 if len({x[1]for x in d})<2:return DTNode([x[1]for x in d][0])
 if not d:return DTNode(__import__("statistics").mode([x[1]for x in d]))
 m,b,p,e=None,None,None,None
 for i in f:
  s,q=partition_by_feature_value(d,i)
  r=sum((len(x)/len(d))*c(x)for x in q)
  if m==None or m>r:m,b,p,e=r,s,q,i
 f.remove(e)
 R=DTNode(b)
 [R.children.append(t(x,c,f)) for x in p]
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
    t = train_tree(dataset, misclassification)
    assert (t.predict((True, False)))
    assert not (t.predict((False, False)))

    training_examples = [
        (("Sunny", "Hot", "High", "Weak"), False),
        (("Sunny", "Hot", "High", "Strong"), False),
        (("Overcast", "Hot", "High", "Weak"), True),
        (("Rain", "Mild", "High", "Weak"), True),
        (("Rain", "Cool", "Normal", "Weak"), True),
        (("Rain", "Cool", "Normal", "Strong"), False),
        (("Overcast", "Cool", "Normal", "Strong"), True),
        (("Sunny", "Mild", "High", "Weak"), False),
        (("Sunny", "Cool", "Normal", "Weak"), True),
        (("Rain", "Mild", "Normal", "Weak"), True),
        (("Sunny", "Mild", "Normal", "Strong"), True),
        (("Overcast", "Mild", "High", "Strong"), True),
        (("Overcast", "Hot", "Normal", "Weak"), True),
        (("Rain", "Mild", "High", "Strong"), False)
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

