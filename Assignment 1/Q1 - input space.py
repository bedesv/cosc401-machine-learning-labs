from itertools import product

def input_space(domains):
    """
        Returns a collection (set, list, …) of tuples. The order of the 
        tuples in the collection is not important. The order of values 
        in each tuple is important; the i-th value should correspond to 
        the i-th attribute and its value should be from the i-th domain.
    """
    return product(*domains)

if __name__ == "__main__":
    domains = [
    {0, 1, 2},
    {True, False},
    ]
    for element in sorted(input_space(domains)):
        print(element)