from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class ShiftGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    @staticmethod
    def generateShiftRight(N):
        program = [
            ('add', [(N+0, '.x'), 1]),
            ('set_pixels', [N+0, N+1, (N+0, '.y'), (N+0, '.c')]),
            ('del', [N+1]),
            ('set_pixels', [N+1, 0, (N+1, '.y'), 0]),
            ('del', [N+1]),
            ('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        return program

    @staticmethod
    def generateShiftUp(N):
        program = []
        program.append(('sub', [(N+0, '.y'), 1]))
        program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
        program.append(('del', [N+1]))
        program.append(('del', [N+0]))

        return program
    
    @staticmethod
    def generateShiftDown(N):
        program = []
        program.append(('add', [(N+0, '.y'), 1]))
        program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, (N+1, '.x'), 0, 0]))
        program.append(('del', [N+1]))
        program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))
        
        return program
    
    @staticmethod
    def generateShiftLeft(N):
        program = [
            ('sub', [(N+0, '.x'), 1]),
            ('set_pixels', [N+0, N+1, (N+0, '.y'), (N+0, '.c')]),
            ('del', [N+1]),
            ('set_pixels', [N+1, (N+0, '.max_x'), (N+1, '.y'), 0]),
            ('del', [N+1]),
            ('del', [N+0])
        ]

        return program   


    def get_task_prog(self):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        if a < 0.25:
            return ShiftGenerator.generateShiftRight(N)
        elif a < 0.5:
            return ShiftGenerator.generateShiftUp(N)
        elif a < 0.75:
            return ShiftGenerator.generateShiftDown(N)
        else:
            return ShiftGenerator.generateShiftLeft(N)

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