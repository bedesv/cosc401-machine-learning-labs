from itertools import permutations, product
from copy import deepcopy
from statistics import mode

placing_points = [20, 16, 14, 12, 10, 9, 8, 7, 6, 5, 4, 3]

top_6 = ["Butchers",
    "Raiders",
    "Drones",
    "Vortex",
    "Silver",
    "Rust"]

bottom_6 = ["Paladins",
    "Monsoon",
    "Steel",
    "Sentinels",
    "Knights",
    "Slackers"]

curr_points = {
    "Butchers": [16, 20],
    "Raiders": [None, 16],
    "Drones": [20, 9],
    "Vortex": [None, 12],
    "Silver": [9, 14],
    "Knights": [12, 8],
    "Rust": [10, 10],
    "Paladins": [14, 7],
    "Monsoon": [None, 6],
    "Steel": [8, 3],
    "Sentinels": [7, 4],
    "Slackers": [6, 5]
}

team_results = {
    "Butchers": [[], [], 12, 1, [], []],
    "Raiders": [[], [], 12, 1, [], []],
    "Drones": [[], [], 12, 1, [], []],
    "Vortex": [[], [], 12, 1, [], []],
    "Silver": [[], [], 12, 1, [], []],
    "Knights": [[], [], 12, 1, [], []],
    "Rust": [[], [], 12, 1, [], []],
    "Paladins": [[], [], 12, 1, [], []],
    "Monsoon": [[], [], 12, 1, [], []],
    "Steel": [[], [], 12, 1, [], []],
    "Sentinels": [[], [], 12, 1, [], []],
    "Slackers": [[], [], 12, 1, [], []]
}

butchers_dont_make_final = set()

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
            if team == "Butchers":
                butchers_dont_make_final.add(tuple(placings[:placing + 1]))
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

with open("results2.txt", 'w') as output_file:
    for team in team_results.keys():


        make_final = team_results[team][0]
        dont_make_final = team_results[team][1]

        output_file.write(f"Team: {team}")

        output_file.write(f"Chances of making final: {len(make_final)} : {len(dont_make_final)}")

        if (make_final):
            output_file.write(f"Average place where make final: {(sum(make_final) / len(make_final)) + 1}")
            output_file.write(f"Lowest place where make final: {max(make_final) + 1}")

        if (dont_make_final):
            output_file.write(f"Average place where don't make final: {(sum(dont_make_final) / len(dont_make_final)) + 1}")
            output_file.write(f"Highest place where don't make final: {min(dont_make_final) + 1}")

        output_file.write(f"Highest possible overall result: {team_results[team][2] + 1}")
        # print("Possible results:")
        # [print(x) for x in team_results[team][4][:10]]

        output_file.write(f"Lowest possible overall result: {team_results[team][3] + 1}")
        # print("Possible results:")
        # [print(x) for x in team_results[team][5][:10]]

        output_file.write("\n")

        [output_file.write(x) for x in butchers_dont_make_final]


