def decode(code):
    min_x = min(code[0], code[2])
    max_x = max(code[0], code[2])
    min_y = min(code[1], code[3])
    max_y = max(code[1], code[3])
    return lambda x: x[0] >= min_x and x[0] <= max_x and x[1] >= min_y and x[1] <= max_y

if __name__ == "__main__":
    import itertools

    h = decode((-1, -1, 1, 1))

    for x in itertools.product(range(-2, 3), repeat=2):
        print(x, h(x))
    