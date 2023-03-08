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

def element_lge(element_a, element_b):
    """
        Takes two elements and returns True if element_a is less general or equal
        to element_b.
    """
    if element_a is None or element_b == "?" or element_a == element_b or (element_a == "?" and element_a == element_b):
        return True
    return False
    
def lge(code_a, code_b):
    """Takes two codes and returns True if code_a is less general or equal
    to code_b."""
    
    # Complete this for the conjunction of constraints. You do not need to
    # decode the given codes.
    for i in range(len(code_a)):
        if not element_lge(code_a[i], code_b[i]):
            return False
    return True
            
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
    output = set()
    code_copy = list(code).copy()
    for i in range(len(code)):
            
        if code[i] is None:
            code_copy[i] = x[i]
        elif code[i] != x[i]:
            code_copy[i] = "?"
    output.add(tuple(code_copy))
    return output

def minimal_specialisations(cc, domains, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal specialisations of the given code with respect
    to the given input x."""
    output = set()

    for i in range(len(cc)):
        for new_element in domains[i] - set(x[i]):
            code_copy = list(cc)
            code_copy[i] = new_element
            output.add(tuple(code_copy))
    
    print("min_spec",output)
    return output


def cea_trace(domains, D):
    S_trace, G_trace = [], []
    S = initial_S(domains)
    G = initial_G(domains)
    # Append S and G (or their copy) to corresponding trace list
    S_trace.append(S)
    G_trace.append(G)
    
    for x, y in D:
        if y: # if positive
            G = {g for g in G if match(g, x)}
            print("sbef",S)
            for s in [s for s in S if not match(s, x)]:
                S.remove(s)
                for h in minimal_generalisations(s, x):
                    
                    for g in G:
                        if (lge(h, g)) and g != h:
                            S.add(h)
                            break
                # S = S.union(minimal_generalisations(s, x))
            new_S = set()
            print("smid",S)
            for s in S:
                more_general = True
                for j in S:
                    if j != s:
                        more_general = more_general and not lge(s, j)
                if more_general:
                    new_S.add(s)
            
            S = new_S
            print("saft",S)

        else: # if negative
            S = {s for s in S if not match(s, x)}
            print("G before", G)
            for g in [g for g in G if match(g, x)]:
                G.remove(g)
                for h in minimal_specialisations(g, domains, x):
                    if h != x:
                        for s in S:
                            if (lge(s, h)) and s != h:
                                G.add(h)
                                break
            print("G mid", G)
            new_G = set()
            for g in G:
                more_specific = True
                for j in S:
                    if g != j:
                        more_specific = more_specific and lge(g, j)
                if more_specific:
                    new_G.add(g)
            
            G = new_G
            print("G after", G)
            

        S_trace.append(S)
        G_trace.append(G)
        # Append S and G (or their copy) to corresponding trace list
    return S_trace, G_trace


if __name__ == "__main__":
    # domains = [
    # {'red', 'blue'}
    # ]

    # training_examples = [
    #     (('red',), True)
    # ]

    # S_trace, G_trace = cea_trace(domains, training_examples)
    # print(len(S_trace), len(G_trace))
    # print(all(type(x) is set for x in S_trace + G_trace))
    # S, G = S_trace[-1], G_trace[-1]
    # print(len(S), len(G))

    # print()
    # domains = [
    #     {'T', 'F'}
    # ]

    # training_examples = []  # no training examples

    # S_trace, G_trace = cea_trace(domains, training_examples)
    # print(len(S_trace), len(G_trace))
    # S, G = S_trace[-1], G_trace[-1]
    # print(len(S), len(G))
    # print()
    # domains = [
    #     ('T', 'F'),
    #     ('T', 'F'),
    # ]

    # training_examples = [
    #     (('F', 'F'), True),
    #     (('T', 'T'), False),
    # ]

    # S_trace, G_trace = cea_trace(domains, training_examples)
    # print(len(S_trace), len(G_trace))
    # S, G = S_trace[-1], G_trace[-1]
    # print(len(S), len(G))

    domains = [
        {"Sunny", "Cloudy", "Rainy"},
        {"Warm", "Cold"},
        {"Normal", "High"},
        {"Strong", "Weak"},
        {"Warm", "Cool"},
        {"Same", "Change"}
    ]

    training_examples = [
        (("Sunny", "Warm", "Normal", "Strong", "Warm", "Same"), True),
        (("Sunny", "Warm", "High", "Strong", "Warm", "Same"), True),
        (("Rainy", "Cold", "High", "Strong", "Warm", "Change"), False),
        (("Sunny", "Warm", "High", "Strong", "Cool", "Change"), True)
    ]

    S_trace, G_trace = cea_trace(domains, training_examples)

    [print(str(s) + "\n") for s in S_trace]

    print("G Trace")
    [print(str(g) + "\n") for g in G_trace]