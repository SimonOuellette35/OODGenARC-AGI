from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class RotationGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    def _generateRot90(self, N):
        program, ref = self.DSL.get_subroutine_rot90(N, N)
        program.append(('set_pixels', [N+0, (N+0, '.y'), (N+0, '.x'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def _generateRot180(self, N):
        program, ref = self.DSL.get_subroutine_rot180(N, N)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def _generateRot270(self, N):
        program, ref = self.DSL.get_subroutine_rot270(N, N)
        program.append(('set_pixels', [N+0, (N+0, '.y'), (N+0, '.x'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def get_task_prog(self):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        if a < 0.333:
            return self._generateRot90(N)
        elif a < 0.666:
            return self._generateRot180(N)
        else:
            return self._generateRot270(N)

    def generate(self, k=1):

        while True:
            # step 1: generate input grid
            input_grid = self.sampler.sample(force_square=True)
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