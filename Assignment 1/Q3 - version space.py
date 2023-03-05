import itertools

def all_possible_functions(X):
    """
        Takes the entire input space (for some problem) and returns a set of functions that is F
    """
    input_list = list(X)
    poss_outputs = [True, False]
    outputs = itertools.product(poss_outputs, repeat=len(X))
    functions = set()
    for output in outputs:
        func = lambda x, output=output: output[input_list.index(x)]
        functions.add(func)
    return functions

def version_space(H, D): 
    """
        Takes a set of hypotheses H, and a training data set D, and returns the version space.
    """
    vs = set()
    for h in H:
        if all(h(x) == y for x, y in D):
            vs.add(h)
    return vs

if __name__ == "__main__":
    X = {"green", "purple"} # an input space with two elements
    D = {("green", True)} # the training data is a subset of X * {True, False}
    F = all_possible_functions(X)
    H = F # H must be a subset of (or equal to) F

    VS = version_space(H, D)

    print(len(VS))

    for h in VS:
        for x, y in D:
            if h(x) != y:
                print("You have a hypothesis in VS that does not agree with the set D!")
                break
        else:
            continue
        break
    else:
        print("OK")    