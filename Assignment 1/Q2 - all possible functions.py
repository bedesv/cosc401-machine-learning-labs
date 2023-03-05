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

if __name__ == "__main__":
    X = {"green", "purple"} # an input space with two elements
    F = all_possible_functions(X)

    # Let's store the image of each function in F as a tuple
    images = set()
    for h in F:
        # print(tuple(h(x) for x in X))
        images.add(tuple(h(x) for x in X))
        
    for image in sorted(images):
        print(image)