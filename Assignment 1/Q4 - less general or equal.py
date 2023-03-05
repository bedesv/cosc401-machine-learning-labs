def less_general_or_equal(ha, hb, X):
    """
        Takes two hypotheses ha and hb, and an input space X and 
        returns True if and only if ha is less general or equal to hb.
    """
    for x in X:
        if ha(x) and not hb(x):
            return False

    return True


if __name__ == "__main__":
    X = list(range(1000))

    def h2(x): return x % 2 == 0
    def h3(x): return x % 3 == 0
    def h6(x): return x % 6 == 0

    H = [h2, h3, h6]

    for ha in H:
        for hb in H:
            print(ha.__name__, "<=", hb.__name__, "?", less_general_or_equal(ha, hb, X))