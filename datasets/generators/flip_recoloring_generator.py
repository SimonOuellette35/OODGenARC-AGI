from ARC_gym.grid_sampling.grid_sampler import GridSampler
from datasets.generators.flip_generator import FlipGenerator
from datasets.generators.recoloring_generator import RecoloringGenerator
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class FlipRecoloringGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

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

        if a < 0.5:
            program = FlipGenerator.generateHFlip(N)
        else:
            program = FlipGenerator.generateVFlip(N)

        a = np.random.uniform()
        if a < 0.333:
            colorB = np.random.choice([c for c in range(10) if c != bg_color])

            tmp_prog = RecoloringGenerator.generateOtherColors(N, bg_color, colorB)

        elif a < 0.666:
            colorA = np.random.choice(unique_colors)
            colorB = np.random.choice([c for c in range(10) if c != colorA])

            tmp_prog = RecoloringGenerator.generateSetColors(N, colorA, colorB)
        else:
            colorA = np.random.choice(unique_colors)
            colorB = np.random.choice([c for c in unique_colors if c != colorA])

            tmp_prog = RecoloringGenerator.generateSwapColor(N, colorA, colorB)

        # combine program and tmp_prog
        program.extend(tmp_prog)

        return program

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