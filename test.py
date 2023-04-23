import ast

results = []
with open("dontmake.txt", "r") as fd:
    results = [ast.literal_eval(line) for line in fd.readlines()]

places = [x.index("Butchers") + 1 for x in results]

print(sum(places) / len(places))

print(len(list(x for x in places if x == 8)))

# [print(results[i]) for i in range(len(places)) if places[i] == 7]

print(min([x.index("Butchers") for x in results if x[0] not in ['Vortex', 'Drones']]))  

# [print(x) for x in results if x[0] not in ['Vortex', 'Drones'] and x[1] != "Drones"]