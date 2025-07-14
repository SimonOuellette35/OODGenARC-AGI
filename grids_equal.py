ground_truth = [
[[1, 4, 7, 1, 2, 3, 4], [3, 2, 8, 4, 1, 8, 4], [4, 1, 1, 1, 7, 8, 4], [1, 1, 2, 3, 8, 1, 3], [1, 1, 1, 1, 4, 7, 3], [4, 4, 1, 1, 4, 3, 4], [2, 8, 1, 3, 2, 4, 1]]
]

attempts = {"00000000": [{"attempt_1": [[2, 8, 1, 3, 2, 4, 1], [2, 8, 1, 3, 2, 4, 1], [2, 8, 1, 3, 2, 4, 1], [1, 1, 1, 1, 4, 7, 3], [1, 1, 1, 1, 4, 7, 3], [1, 1, 1, 1, 4, 7, 3], [1, 1, 1, 1, 4, 7, 3]], "attempt_2": [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 4], [1, 1, 1, 1, 1, 1, 3], [2, 2, 2, 2, 2, 3, 3], [1, 1, 1, 1, 1, 1, 4], [8, 8, 8, 8, 8, 4, 4], [7, 7, 7, 7, 7, 1, 4]], "attempt_3": [[3, 2, 8, 4, 1, 8, 4], [3, 2, 8, 4, 1, 8, 4], [3, 2, 8, 4, 1, 8, 4], [3, 2, 8, 4, 1, 8, 4], [3, 2, 8, 4, 1, 8, 4], [3, 2, 8, 4, 1, 8, 4], [4, 4, 4, 4, 3, 4, 4]], "attempt_4": [[1, 8, 8, 8, 8, 8, 8], [4, 4, 4, 4, 4, 4, 4], [3, 1, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1, 1], [4, 2, 2, 2, 2, 2, 2], [4, 4, 4, 4, 4, 4, 4]], "attempt_5": [[4, 4, 1, 1, 4, 3, 4], [4, 4, 1, 1, 4, 3, 4], [4, 4, 1, 1, 4, 3, 4], [4, 4, 1, 1, 4, 3, 4], [4, 4, 1, 1, 4, 3, 4], [4, 4, 1, 1, 4, 3, 4], [4, 4, 1, 1, 4, 3, 4]], "attempt_6": [[4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4], [4, 4, 4, 4, 4, 4, 4]], "attempt_7": [[1, 1, 1, 1, 4, 7, 3], [1, 1, 1, 1, 4, 7, 3], [1, 1, 1, 1, 4, 7, 3], [1, 1, 2, 3, 8, 1, 3], [1, 4, 7, 1, 2, 3, 4], [1, 4, 7, 1, 2, 3, 4], [1, 4, 7, 1, 2, 3, 4]], "attempt_8": [[3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3, 3]]}]}


# Find the most frequent non-empty grid attempt
attempt_grids = attempts[list(attempts.keys())[0]][0]
grid_counts = {}

# Count occurrences of each non-empty grid
for attempt_name, grid in attempt_grids.items():
    if grid:  # Check if grid is not empty
        grid_tuple = tuple(tuple(row) for row in grid)  # Convert to tuple for hashing
        if grid_tuple in grid_counts:
            grid_counts[grid_tuple] += 1
        else:
            grid_counts[grid_tuple] = 1

# Find the most frequent non-empty grid
most_frequent_grid = None
max_count = 0
for grid_tuple, count in grid_counts.items():
    if count > max_count:
        max_count = count
        most_frequent_grid = list(list(row) for row in grid_tuple)

# If all grids have same frequency, pick the first one
if most_frequent_grid is None:
    print("==> ALL DISTINCT GRIDS! Defaulting to first one.")
    most_frequent_grid = list(attempt_grids.values())[0]

print("Most frequent result grid:")
for row in most_frequent_grid:
    print(row)

print("\nGround truth grid:")
for row in ground_truth:
    print(row)

if ground_truth == most_frequent_grid:
    print("\nThe grids are exactly the same!")
else:
    print("\nThe grids are different!")
