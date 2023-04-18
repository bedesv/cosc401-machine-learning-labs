from itertools import permutations
from copy import deepcopy
from statistics import mode

placing_points = [20, 16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3]

curr_points = {
    "Butchers": [16, 20],
    "Raiders": [None, 16],
    "Drones": [20, 9],
    "Vortex": [None, 12],
    "Silver": [9, 14],
    "Knights": [14, 8],
    "Rust": [10, 10],
    "Paladins": [12, 7],
    # "Monsoon": [None, 6],
    # "Steel": [8, 3],
    # "Sentinels": [7, 4],
    # "Slackers": [6, 5]
}

make_final = []
dont_make_final = []

fifth_dont_make = set()
eigth = set()

print("Finding Permutations")
possible_placings = list(permutations(list(curr_points.keys())))
print(f"{len(possible_placings)} Permutations Found")


index = 1
for placings in possible_placings:
    print(f"{round(index/len(possible_placings) * 100, 2)}% completed", end="\r")
    index += 1
    temp_points = deepcopy(curr_points)
    for i in range(len(placings)):
        temp_points[placings[i]].append(placing_points[i])
        if temp_points[placings[i]][0] is None:
            temp_points[placings[i]][0] = int(sum(temp_points[placings[i]][1:]) / 2)
        
        temp_points[placings[i]].append(sum(temp_points[placings[i]]))
    result = sorted(temp_points.keys(), key=lambda x: sum(temp_points[x]) / 2, reverse=True)
    # print(result)
    if "Butchers" not in result[:2]:
        # dont_make_final.append(placings)
        dont_make_final.append(placings.index("Butchers"))
        if placings.index("Butchers") == 4:
            fifth_dont_make.add(placings[:5])
    else:
        
        # make_final.append(placings)
        make_final.append(placings.index("Butchers"))
        if placings.index("Butchers") == 7 and placings[1] == "Vortex":
            print(placings)
            print(temp_points)
            print()

print("Make final : Don't make final")
print(f"{len(make_final)} : {len(dont_make_final)}")

# for placing in dont_make_final:
#     print(placing)

# winners = [x[0] for x in dont_make_final if x[0] == "Vortex"]
# print(len(winners))
# print(mode(winners))

print(sum(make_final) / len(make_final))
print(max(make_final))
print(sum(dont_make_final) / len(dont_make_final))
print(min(dont_make_final))

# [print(x) for x in fifth_dont_make]
[print(x) for x in eigth]

# for result in output_points:
#     for team, points in result.items():
#         print(f"{team}: {sum(points)}")
#     print()

# for i in range(len(curr_points)):
#     for place in range(len(placing_points)):
