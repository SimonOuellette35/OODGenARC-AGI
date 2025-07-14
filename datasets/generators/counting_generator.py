from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class CountingGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    
    @staticmethod
    def generateMaxCountFill(N, dim):
        # Count instances per color. 
        # Get the color with maximum count.
        # Draw a 1x1, 2x2, 3x3, 4x4 or 5x5 grid filled with this color.
        program = [
            ('color_set', [N+0]),
            ('count_values', [N+1, (N+0, '.c')]),
            ('arg_max', [N+1, N+2]),
            ('del', [N+1]),
            ('del', [N+1]),
            ('new_grid', [dim, dim, N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        return program

    @staticmethod
    def generateMinCountFill(N, dim):
        # Count instances per color. 
        # Get the color with minimum count.
        # Draw a 1x1, 2x2, 3x3, 4x4 or 5x5 grid filled with this color.
        program = [
            ('color_set', [N+0]),
            ('count_values', [N+1, (N+0, '.c')]),
            ('arg_min', [N+1, N+2]),
            ('del', [N+1]),
            ('del', [N+1]),
            ('new_grid', [dim, dim, N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        return program

    @staticmethod
    def generateCountAndDrawH(N, bg_color):
        # Count how many non-bg pixels there are. 
        # Draw horizontally that many pixels of the same color, as the output grid.
        program = [
            # 1) get the foreground color
            ('color_set', [N+0]),
            ('equal', [N+1, bg_color]),
            ('exclude', [N+1, N+2]),
            ('del', [N+1]),
            ('del', [N+1]),

            # 2) count_values for this foreground color
            ('count_values', [N+1, (N+0, '.c')]),

            # 3) create the output grid
            ('new_grid', [N+2, 1, N+1]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        return program

    @staticmethod
    def generateCountAndDrawV(N, bg_color):
        # Count how many non-bg pixels there are. 
        # Draw vertically that many pixels of the same color, as the output grid.
        program = [
            # 1) get the foreground color
            ('color_set', [N+0]),
            ('equal', [N+1, bg_color]),
            ('exclude', [N+1, N+2]),
            ('del', [N+1]),
            ('del', [N+1]),

            # 2) count_values for this foreground color
            ('count_values', [N+1, (N+0, '.c')]),

            # 3) create the output grid
            ('new_grid', [1, N+2, N+1]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        return program

    def get_task_prog(self):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        if a < 0.25:
            valid = False
            while not valid:
                input_grid = self.sampler.sample(min_dim=3, max_dim=5, monochrome_grid_ok=False)

                # Count occurrences of each color in the input grid
                color_counts = {}
                for row in input_grid:
                    for color in row:
                        color_counts[color] = color_counts.get(color, 0) + 1
                
                # Find the maximum count
                max_count = max(color_counts.values())
                
                # Check if maximum is unique by counting how many colors have this count
                colors_with_max = sum(1 for count in color_counts.values() if count == max_count)
                valid = colors_with_max == 1
        
            dim = np.random.choice(np.arange(1, 6))
            if len(input_grid) > 5 or len(input_grid[0]) > 5:
                print("ERROR: shouldn't create grid of dim > 5!")
                exit(-1)
            return CountingGenerator.generateMaxCountFill(N, dim), input_grid
        elif a < 0.5:
            valid = False
            while not valid:
                input_grid = self.sampler.sample(min_dim=3, max_dim=5, monochrome_grid_ok=False)
                # Count occurrences of each color in the input grid
                color_counts = {}
                for row in input_grid:
                    for color in row:
                        color_counts[color] = color_counts.get(color, 0) + 1
                
                # Find the minimum count
                min_count = min(color_counts.values())
                
                # Check if minimum is unique by counting how many colors have this count
                colors_with_min = sum(1 for count in color_counts.values() if count == min_count)
                valid = colors_with_min == 1
                
            dim = np.random.choice(np.arange(1, 6))
            if len(input_grid) > 5 or len(input_grid[0]) > 5:
                print("ERROR: shouldn't create grid of dim > 5!")
                exit(-1)
            return CountingGenerator.generateMinCountFill(N, dim), input_grid
        elif a < 0.75:
            # Generate a 3x3 grid with random background color
            bg_color = np.random.randint(0, 10)
            input_grid = np.full((3, 3), bg_color)

            # Pick two different foreground colors
            possible_colors = [c for c in range(10) if c != bg_color]
            fg_colors = np.random.choice(possible_colors, size=2, replace=False)

            # Pick number of pixels for each foreground color (1-4 each, ensuring at least 2 colors)
            num_pixels1 = np.random.randint(1, 5)
            num_pixels2 = np.random.randint(1, 5)

            # Randomly place the first foreground color pixels
            positions1 = np.random.choice(9, num_pixels1, replace=False)
            for pos in positions1:
                row = pos // 3
                col = pos % 3
                input_grid[row, col] = fg_colors[0]

            # Randomly place the second foreground color pixels in remaining positions
            remaining_positions = [p for p in range(9) if p not in positions1]
            positions2 = np.random.choice(remaining_positions, min(num_pixels2, len(remaining_positions)), replace=False)
            for pos in positions2:
                row = pos // 3
                col = pos % 3
                input_grid[row, col] = fg_colors[1]

            if len(input_grid) > 5 or len(input_grid[0]) > 5:
                print("ERROR: shouldn't create grid of dim > 5!")
                exit(-1)

            # Check if there are at least 2 unique colors in the input grid
            unique_colors = np.unique(input_grid)
            if len(unique_colors) < 2:
                print("ERROR: Input grid must have at least 2 unique colors!")
                exit(-1)
            return CountingGenerator.generateCountAndDrawH(N, bg_color), input_grid
        else:
            # Generate a 3x3 grid with random background color
            bg_color = np.random.randint(0, 10)
            input_grid = np.full((3, 3), bg_color)

            # Pick two different foreground colors
            possible_colors = [c for c in range(10) if c != bg_color]
            fg_colors = np.random.choice(possible_colors, size=2, replace=False)

            # Pick number of pixels for each foreground color (1-4 each, ensuring at least 2 colors)
            num_pixels1 = np.random.randint(1, 5)
            num_pixels2 = np.random.randint(1, 5)

            # Randomly place the first foreground color pixels
            positions1 = np.random.choice(9, num_pixels1, replace=False)
            for pos in positions1:
                row = pos // 3
                col = pos % 3
                input_grid[row, col] = fg_colors[0]

            # Randomly place the second foreground color pixels in remaining positions
            remaining_positions = [p for p in range(9) if p not in positions1]
            positions2 = np.random.choice(remaining_positions, min(num_pixels2, len(remaining_positions)), replace=False)
            for pos in positions2:
                row = pos // 3
                col = pos % 3
                input_grid[row, col] = fg_colors[1]

            if len(input_grid) > 5 or len(input_grid[0]) > 5:
                print("ERROR: shouldn't create grid of dim > 5!")
                exit(-1)

            # Check if there are at least 2 unique colors in the input grid
            unique_colors = np.unique(input_grid)
            if len(unique_colors) < 2:
                print("ERROR: Input grid must have at least 2 unique colors!")
                exit(-1)
            return CountingGenerator.generateCountAndDrawV(N, bg_color), input_grid

    def generate(self, k=1):

        # step 2: prog ground truth
        prog, input_grid = self.get_task_prog()

        input_grid_DSL = self.DSL.Grid(input_grid)
        initial_state = [input_grid_DSL]

        # step 3: use prog to generate target_grid
        token_seq_list = ProgUtils.convert_prog_to_token_seq(prog, self.DSL)
        output_grid_DSL = pi.execute(token_seq_list, initial_state, self.DSL)

        return input_grid, output_grid_DSL.cells, prog