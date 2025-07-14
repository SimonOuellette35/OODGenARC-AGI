from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np
import random


class CroppingGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    @staticmethod
    def generateCropUL(N, dim):
        program = [
            ('crop', [N+0, 0, 0, dim, dim]),
            ('del', [N+0])
        ]
        return program

    @staticmethod
    def generateCropUR(N, dim):
        program = [
            ('sub', [(N+0, '.width'), dim]),
            ('crop', [N+0, N+1, 0, (N+0, '.width'), dim]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    @staticmethod
    def generateCropLL(N, dim):
        program = [
            ('sub', [(N+0, '.height'), dim]),
            ('crop', [N+0, 0, N+1, dim, (N+0, '.height')]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    @staticmethod
    def generateCropLR(N, dim):
        program = [
            ('sub', [(N+0, '.width'), dim]),
            ('sub', [(N+0, '.height'), dim]),
            ('crop', [N+0, N+1, N+2, (N+0, '.width'), (N+0, '.height')]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    @staticmethod
    def generateCropInside(N, margin):
        program = [
            ('sub', [(N+0, '.width'), margin]),
            ('sub', [(N+0, '.height'), margin]),
            ('crop', [N+0, margin, margin, N+1, N+2]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    def get_task_prog(self, input_grid):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        dim = np.random.randint(2, 6)
        # Generate all possible programs
        ul_prog = CroppingGenerator.generateCropUL(N, dim)
        ur_prog = CroppingGenerator.generateCropUR(N, dim)
        ll_prog = CroppingGenerator.generateCropLL(N, dim)
        lr_prog = CroppingGenerator.generateCropLR(N, dim)
        max_margin = min((input_grid.shape[0] // 2) - 1, (input_grid.shape[1] // 2) - 1)
        margin = np.random.randint(1, min(4, max_margin + 1))
        inside_prog = CroppingGenerator.generateCropInside(N, margin)

        # Select program based on a
        if a < 0.2:
            selected_prog = ul_prog
        elif a < 0.4:
            selected_prog = ur_prog
        elif a < 0.6:
            selected_prog = ll_prog
        elif a < 0.8:
            selected_prog = lr_prog
        else:
            selected_prog = inside_prog

        other_progs = []
        if selected_prog != ul_prog:
            other_progs.append(ul_prog)
        if selected_prog != ur_prog:
            other_progs.append(ur_prog)
        if selected_prog != ll_prog:
            other_progs.append(ll_prog)
        if selected_prog != lr_prog:
            other_progs.append(lr_prog)
        if selected_prog != inside_prog:
            other_progs.append(inside_prog)

        return selected_prog, other_progs

    def generate(self, k=1):

        while True:
            # step 1: generate input grid
            input_grid = self.sampler.sample(min_dim=6)
            input_grid_DSL = self.DSL.Grid(input_grid)
            initial_state = [input_grid_DSL]

            # step 2: prog ground truth
            prog, other_progs = self.get_task_prog(input_grid)

            # step 3: use prog to generate target_grid
            token_seq_list = ProgUtils.convert_prog_to_token_seq(prog, self.DSL)
            output_grid_DSL = pi.execute(token_seq_list, initial_state, self.DSL)

            # Generate output grids for all other programs
            other_outputs = []
            for other_prog in other_progs:
                token_seq_list = ProgUtils.convert_prog_to_token_seq(other_prog, self.DSL)
                initial_state = [input_grid_DSL]
                other_output = pi.execute(token_seq_list, initial_state, self.DSL)
                other_outputs.append(other_output.cells)

            # Check if output grid matches any other outputs
            # the following code is horrible but for some reason a normal comparison fails...
            valid = True
            output_cells = np.array([list(row) for row in output_grid_DSL.cells]).astype(int)
            for other_output in other_outputs:
                other_cells = np.array([list(row) for row in other_output]).astype(int)

                if np.array_equal(output_cells, other_cells):
                    valid = False
                    
            if not valid:
                continue

            # Check if input and output grids are different and output is not all black
            cells_int = np.array([list(row) for row in output_grid_DSL.cells]).astype(int)
            if not np.array_equal(input_grid, output_grid_DSL.cells) and not np.all(cells_int == 0):
                return input_grid, output_grid_DSL.cells, prog