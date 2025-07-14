from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class RecoloringGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    # this switches colors other than exclude_color to target_color
    @staticmethod
    def generateOtherColors(N, exclude_color, target_color):
        program = [
            ('equal', [(N+0, '.c'), exclude_color]),
            ('switch', [N+1, exclude_color, target_color]),
            ('del', [N+1]),
            ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    # this switches colors source_color to target_color
    @staticmethod
    def generateSetColors(N, source_color, target_color):
        program = [
            ('equal', [(N+0, '.c'), source_color]),
            ('switch', [N+1, target_color, (N+0, '.c')]),
            ('del', [N+1]),
            ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    # this swaps two colors (all pixels of color A are set to B, and all pixels of color B are set to A)
    @staticmethod
    def generateSwapColor(N, color_a, color_b):
        program = [
            ('equal', [(N+0, '.c'), color_a]),
            ('equal', [(N+0, '.c'), color_b]),
            ('switch', [[N+1, N+2], [color_b, color_a], (N+0, '.c')]),
            ('del', [N+1]),
            ('del', [N+1]),
            ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    def get_task_prog(self, bg_color, input_grid):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        # Get unique colors in input grid
        unique_colors = set()
        for row in input_grid:
            for color in row:
                unique_colors.add(color)

        unique_colors = list(unique_colors)

        if len(unique_colors) <= 2:
            return None
        
        if a < 0.333:
            colorB = np.random.choice([c for c in range(10) if c != bg_color])

            return RecoloringGenerator.generateOtherColors(N, bg_color, colorB)
        elif a < 0.666:
            colorA = np.random.choice(unique_colors)
            colorB = np.random.choice([c for c in range(10) if c != colorA])

            return RecoloringGenerator.generateSetColors(N, colorA, colorB)
        else:
            colorA = np.random.choice(unique_colors)
            colorB = np.random.choice([c for c in unique_colors if c != colorA])

            return RecoloringGenerator.generateSwapColor(N, colorA, colorB)


    def generate(self, k=1):

        while True:
            # step 1: generate input grid
            input_grid = self.sampler.sample(monochrome_grid_ok=False)
            input_grid_DSL = self.DSL.Grid(input_grid)
            initial_state = [input_grid_DSL]

            # Count occurrences of each color in the input grid
            color_counts = {}
            for row in input_grid:
                for color in row:
                    color_counts[color] = color_counts.get(color, 0) + 1
            
            # Find the most common color
            bg_color = max(color_counts.items(), key=lambda x: x[1])[0]

            # step 2: prog ground truth
            prog = self.get_task_prog(bg_color, input_grid)
            if prog is None:
                continue

            # step 3: use prog to generate target_grid
            token_seq_list = ProgUtils.convert_prog_to_token_seq(prog, self.DSL)
            output_grid_DSL = pi.execute(token_seq_list, initial_state, self.DSL)

            return input_grid, output_grid_DSL.cells, prog