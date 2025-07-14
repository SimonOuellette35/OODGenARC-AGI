from ARC_gym.grid_sampling.grid_sampler import GridSampler
from datasets.generators.flip_generator import FlipGenerator
from datasets.generators.shift_generator import ShiftGenerator
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class FlipShiftGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    def get_task_prog(self):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        if a < 0.5:
            program = FlipGenerator.generateHFlip(N)
        else:
            program = FlipGenerator.generateVFlip(N)

        a = np.random.uniform()
        if a < 0.25:
            tmp_prog = ShiftGenerator.generateShiftDown(N)
        elif a < 0.5:
            tmp_prog = ShiftGenerator.generateShiftUp(N)
        elif a < 0.75:
            tmp_prog = ShiftGenerator.generateShiftLeft(N)
        else:
            tmp_prog = ShiftGenerator.generateShiftRight(N)

        # combine program and tmp_prog
        program.extend(tmp_prog)

        return program

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

            # Check if output grid dimensions are within limits
            if len(output_grid_DSL.cells) <= 30 and len(output_grid_DSL.cells[0]) <= 30:
                return input_grid, output_grid_DSL.cells, prog