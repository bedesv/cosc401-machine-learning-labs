from itertools import permutations, product
from copy import deepcopy
from statistics import mode

placing_points = [20, 16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3]

top_6 = ["Black",
    "Valhalla",
    "Hornets",
    "Expose",
    "Hurricanes",
    "Justice"]

bottom_6 = ["Wasps",
    "Gold",
    "Bronze",
    "TKT",
    "Bailiffs",
    "Jotnar"]

curr_points = {
    "Black": [16, 20],
    "Valhalla": [14, 16],
    "Hornets": [20, 14],
    "Expose": [12, 12],
    "Hurricanes": [10, 10],
    "Justice": [9, 9],
    "Wasps": [6, 8],
    "Gold": [8, 7],
    "Bronze": [3, 6],
    "TKT": [7, 5],
    "Bailiffs": [4, 4],
    "Jotnar": [5, 3]
}

team_results = dict()

[team_results.__setitem__(key, [[], [], 12, 1, [], []]) for key in curr_points.keys()]

# {
#     "Butchers": [[], [], 12, 1, [], []],
#     "Raiders": [[], [], 12, 1, [], []],
#     "Drones": [[], [], 12, 1, [], []],
#     "Vortex": [[], [], 12, 1, [], []],
#     "Silver": [[], [], 12, 1, [], []],
#     "Knights": [[], [], 12, 1, [], []],
#     "Rust": [[], [], 12, 1, [], []],
#     "Paladins": [[], [], 12, 1, [], []],
#     "Monsoon": [[], [], 12, 1, [], []],
#     "Steel": [[], [], 12, 1, [], []],
#     "Sentinels": [[], [], 12, 1, [], []],
#     "Slackers": [[], [], 12, 1, [], []]
# }

# butchers_dont_make_final = set()

print("Finding Permutations")

top_4 = list(permutations(top_6, 4))
middle_4 = list(permutations(list(curr_points.keys()), 4))
bottom_4 = list(permutations(bottom_6, 4))


def placings_generator(top, middle, bottom):
    for t in top:
        for m in middle:
            for b in bottom:
                placing = t + m + b
                if len(set(placing)) == 12:
                    yield tuple(placing)

possible_placings = placings_generator(top_4, middle_4, bottom_4)


index = 1
for placings in possible_placings:
    print(f"{index} iterations completed", end="\r")
    index += 1
    temp_points = deepcopy(curr_points)
    for i in range(len(placings)):
        temp_points[placings[i]].append(placing_points[i])
        if temp_points[placings[i]][0] is None:
            temp_points[placings[i]][0] = int(sum(temp_points[placings[i]][1:]) / 2)
        
        temp_points[placings[i]].append(sum(temp_points[placings[i]]))
    result = sorted(temp_points.keys(), key=lambda x: sum(temp_points[x]) / 2, reverse=True)
    
    for team in team_results.keys():

        placing = placings.index(team)
        overall_placing = result.index(team)

        if team not in result[:2]:
            team_results[team][1].append(placing)
            # if team == "Justice":
            #     butchers_dont_make_final.add(tuple(placings[:placing + 1]))
        else:
            team_results[team][0].append(placing)

        # Best result
        if team_results[team][2] > overall_placing:
            team_results[team][2] = overall_placing
            team_results[team][4] = [placings]
        elif team_results[team][2] == overall_placing:
            team_results[team][4].append(placings)
        
        # Worst result
        if team_results[team][3] < overall_placing:
            team_results[team][3] = overall_placing
            team_results[team][5] = [placings]
        elif team_results[team][3] == overall_placing:
            team_results[team][5].append(placings)

print("Saving output")

with open("results3.txt", 'w') as output_file:
    for team in team_results.keys():


        make_final = team_results[team][0]
        dont_make_final = team_results[team][1]

        output_file.write(f"Team: {team}\n")

        output_file.write(f"Chances of making final: {len(make_final)} : {len(dont_make_final)}\n")

        if (make_final):
            output_file.write(f"Average place where make final: {(sum(make_final) / len(make_final)) + 1}\n")
            output_file.write(f"Lowest place where make final: {max(make_final) + 1}\n")

        if (dont_make_final):
            output_file.write(f"Average place where don't make final: {(sum(dont_make_final) / len(dont_make_final)) + 1}\n")
            output_file.write(f"Highest place where don't make final: {min(dont_make_final) + 1}\n")

        output_file.write(f"Highest possible overall result: {team_results[team][2] + 1}\n")
        # print("Possible results:")
        # [print(x) for x in team_results[team][4][:10]]

        output_file.write(f"Lowest possible overall result: {team_results[team][3] + 1}\n")
        # print("Possible results:")
        # [print(x) for x in team_results[team][5][:10]]

        output_file.write("\n")

    # [output_file.write(f"{x}\n") for x in butchers_dont_make_final]


