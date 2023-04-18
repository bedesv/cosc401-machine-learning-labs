import unittest
from code import DTNode, partition_by_feature_value, misclassification, gini, entropy, train_tree

class TestQuestion1(unittest.TestCase):

    def test_1(self):
        node = DTNode(True) 
        x = (1, 2, 3)
        self.assertTrue(node.predict(x))
        self.assertTrue(node.predict(None))

    def test_2(self):
        yes_node = DTNode("Yes")
        no_node = DTNode("No")
        tree_root = DTNode(lambda x: 0 if x[2] < 4 else 1)
        tree_root.children = [yes_node, no_node]

        self.assertEqual(tree_root.predict((False, 'Red', 3.5)), "Yes")
        self.assertEqual(tree_root.predict((False, 'Green', 6.1)), "No")


class TestQuestion2(unittest.TestCase):

    def test_1(self):
        n = DTNode(True)
        self.assertEqual(n.leaves(), 1)

    def test_2(self):
        t = DTNode(True)
        f = DTNode(False)
        n = DTNode(lambda v: 0 if not v else 1)
        n.children = [t, f]
        self.assertEqual(n.leaves(), 2)


class TestQuestion3(unittest.TestCase):

    def test_1(self):
        dataset = [
            ((True, True), False),
            ((True, False), True),
            ((False, True), True),
            ((False, False), False),
        ]
        f, p = partition_by_feature_value(dataset,  0)
        p = sorted(sorted(partition) for partition in p)
        expected_p = [[((False, False), False), ((False, True), True)],
                        [((True, False), True), ((True, True), False)]]
        self.assertEqual(p, expected_p)
        partition_index = f((True, True))
        # Everything in the "True" partition for feature 0 is true
        self.assertTrue(all(x[0]==True for x,c in p[partition_index]))
        partition_index = f((False, True))
        # Everything in the "False" partition for feature 0 is false
        self.assertTrue(all(x[0]==False for x,c in p[partition_index]))

    def test_2(self):
        dataset = [
            (("a", "x", 2), False),
            (("b", "x", 2), False),
            (("a", "y", 5), True),
        ]
        f, p = partition_by_feature_value(dataset, 1)
        p = (sorted(sorted(partition) for partition in p))
        expected_p = [[(('a', 'x', 2), False), (('b', 'x', 2), False)], [(('a', 'y', 5), True)]]
        self.assertEqual(expected_p, p)
        partition_index = f(("a", "y", 5))
        # everything in the "y" partition for feature 1 has a y
        self.assertTrue(all(x[1]=="y" for x, c in p[partition_index]))


class TestQuestion4(unittest.TestCase):

    dataset = [
        ((False, False), False),
        ((False, True), True),
        ((True, False), True),
        ((True, True), False)
    ]

    def test_misclassification(self):
        actual = round(misclassification(self.dataset), 4)
        self.assertEqual(actual, 0.5000)

    def test_gini(self):
        actual = round(gini(self.dataset), 4)
        self.assertEqual(actual, 0.5000)
    
    def test_entropy(self):
        actual = round(entropy(self.dataset), 4)
        self.assertEqual(actual, 1.0000)


class TestQuestion5(unittest.TestCase):

    def test_1(self):
        dataset = [
            ((True, True), False),
            ((True, False), True),
            ((False, True), True),
            ((False, False), False)
        ]
        t = train_tree(dataset, misclassification)
        self.assertTrue(t.predict((True, False)))
        self.assertFalse(t.predict((False, False)))

    def test_2(self):
        dataset = [
            (("Sunny",    "Hot",  "High",   "Weak"),   False),
            (("Sunny",    "Hot",  "High",   "Strong"), False),
            (("Overcast", "Hot",  "High",   "Weak"),   True),
            (("Rain",     "Mild", "High",   "Weak"),   True),
            (("Rain",     "Cool", "Normal", "Weak"),   True),
            (("Rain",     "Cool", "Normal", "Strong"), False), #
            (("Overcast", "Cool", "Normal", "Strong"), True),
            (("Sunny",    "Mild", "High",   "Weak"),   False),
            (("Sunny",    "Cool", "Normal", "Weak"),   True), #
            (("Rain",     "Mild", "Normal", "Weak"),   True),
            (("Sunny",    "Mild", "Normal", "Strong"), True), #
            (("Overcast", "Mild", "High",   "Strong"), True),
            (("Overcast", "Hot",  "Normal", "Weak"),   True),
            (("Rain",     "Mild", "High",   "Strong"), False), #
        ]
        t = train_tree(dataset, misclassification)
        self.assertTrue(t.predict(("Overcast", "Cool", "Normal", "Strong")))
        self.assertTrue(t.predict(("Sunny", "Cool", "Normal", "Strong")))
        for i, values in enumerate(dataset):
            vector, output = values
            print(i, t.predict(vector) == output)
            self.assertEqual(t.predict(vector), output)


if __name__ == '__main__':
    unittest.main()