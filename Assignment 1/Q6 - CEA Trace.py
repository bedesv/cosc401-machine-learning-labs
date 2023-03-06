from itertools import product

def input_space(domains):
    """
        Returns a collection (set, list, …) of tuples. The order of the 
        tuples in the collection is not important. The order of values 
        in each tuple is important; the i-th value should correspond to 
        the i-th attribute and its value should be from the i-th domain.
    """
    return product(*domains)

def decode(code):
    """Takes a code and returns the corresponding hypothesis."""
    def h(x):
        return len(x) == len(code) and all([code[i] is not None and (code[i] == "?" or code[i] == x[i]) for i in range(len(code))])
    return h
        
def match(code, x):
    """Takes a code and returns True if the corresponding hypothesis returns
    True (positive) for the given input."""
    return decode(code)(x)
    
def lge(code_a, code_b):
    """Takes two codes and returns True if code_a is less general or equal
    to code_b."""
    
    # Complete this for the conjunction of constraints. You do not need to
    # decode the given codes.
    for i in range(len(code_a)):
        pass


    

            
def initial_S(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of S."""
    
    result = set()
    result.add(tuple(None for _ in domains))
    return result

    
def initial_G(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of G."""
    result = set()
    result.add(tuple("?" for _ in domains))
    return result


def minimal_generalisations(code, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal generalisations of the given code with respect
    to the given input x."""
    
    # Return an appropriate set


def minimal_specialisations(cc, domains, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal specialisations of the given code with respect
    to the given input x."""
    
    # Return an appropriate set


def cea_trace(domains, D):
    S_trace, G_trace = [], []
    S = initial_S(domains)
    G = initial_G(domains)
    # Append S and G (or their copy) to corresponding trace list

    print(G)
    print(S)
    
    for x, y in D:
        if y: # if positive
            pass
            # Complete
            
        else: # if negative
            pass
            # Complete

        # Append S and G (or their copy) to corresponding trace list

    return S_trace, G_trace


if __name__ == "__main__":
    domains = [
    {'red', 'blue'}
    ]

    training_examples = [
        (('red',), True)
    ]

    S_trace, G_trace = cea_trace(domains, training_examples)
    print(len(S_trace), len(G_trace))
    print(all(type(x) is set for x in S_trace + G_trace))
    S, G = S_trace[-1], G_trace[-1]
    print(len(S), len(G))