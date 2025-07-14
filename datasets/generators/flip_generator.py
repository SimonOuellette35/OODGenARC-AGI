from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class FlipGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    @staticmethod
    def generateHFlip(N, i=0):
        program = [
            ('sub', [(N+i, '.max_x'), (N+i, '.x')]),
            ('set_pixels', [N+i, N+i+1, (N+i, '.y'), (N+i, '.c')]),
            ('del', [N+i]),
            ('del', [N+i])
        ]
        return program

    @staticmethod
    def generateVFlip(N, i=0):
        program = [
            ('sub', [(N+i, '.max_y'), (N+i, '.y')]),
            ('set_pixels', [N+i, (N+i, '.x'), N+i+1, (N+i, '.c')]),
            ('del', [N+i]),
            ('del', [N+i])
        ]

        return program

    def get_task_prog(self):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        if a < 0.5:
            return FlipGenerator.generateHFlip(N)
        else:
            return FlipGenerator.generateVFlip(N)

    def generate(self, k=1):

        while True:
            # step 1: generate input grid
            input_grid = self.sampler.sample()
            input_grid_DSL = self.DSL.Grid(input_grid)
            initial_state = [input_grid_DSL]

            # step 2: prog ground truth
            prog = self.get_task_prog()

            # step 3: use prog to generate target_grid
            token_seq_list = ProgUtils.convert_prog_to_token_seq(prog, self.DSL)
            output_grid_DSL = pi.execute(token_seq_list, initial_state, self.DSL)

            # Check if input and output grids are different
            if not np.array_equal(input_grid, output_grid_DSL.cells):
                return input_grid, output_grid_DSL.cells, prog