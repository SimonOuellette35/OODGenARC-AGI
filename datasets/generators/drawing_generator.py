from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class DrawingGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    
    @staticmethod
    def generateDrawLine(N, where, color):

        if where == 'top':
            program = [
                ('set_pixels', [N+0, (N+0, '.x'), 0, color]),
                ('del', [N+0])
            ]

        elif where == 'bottom':
            program = [
                ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.max_y'), color]),
                ('del', [N+0])
            ]

        elif where == 'left_col':
            program = [
                ('set_pixels', [N+0, 0, (N+0, '.y'), color]),
                ('del', [N+0])
            ]

        elif where == 'right_col':
            program = [
                ('set_pixels', [N+0, (N+0, '.max_x'), (N+0, '.y'), color]),
                ('del', [N+0])
            ]

        elif where == 'center_vertical':
            program = [
                ('div', [(N+0, '.width'), 2]),
                ('set_pixels', [N+0, N+1, (N+0, '.y'), color]),
                ('del', [N+0]),
                ('del', [N+0])
            ]

        elif where == 'center_horizontal':
            program = [
                ('div', [(N+0, '.height'), 2]),
                ('set_pixels', [N+0, (N+0, '.x'), N+1, color]),
                ('del', [N+0]),
                ('del', [N+0])
            ]

        return program

    @staticmethod
    def generateDrawContour(N, color):
        program = [
            ('set_pixels', [N+0, 0, (N+0, '.y'), color]),
            ('del', [N+0]),
            ('set_pixels', [N+0, (N+0, '.max_x'), (N+0, '.y'), color]),
            ('del', [N+0]),
            ('set_pixels', [N+0, (N+0, '.x'), 0, color]),
            ('del', [N+0]),
            ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.max_y'), color]),
            ('del', [N+0])
        ]

        return program

    @staticmethod
    def generateDrawGrid(N, color):
        program = [
            ('mod', [(N+0, '.x'), 2]),
            ('mod', [(N+0, '.y'), 2]),
            ('equal', [N+1, 0]),
            ('equal', [N+2, 0]),
            ('or', [N+3, N+4]),
            ('del', [N+1]),
            ('del', [N+1]),
            ('del', [N+1]),
            ('del', [N+1]),
            ('keep', [(N+0, '.x'), N+1]),
            ('keep', [(N+0, '.y'), N+1]),
            ('del', [N+1]),
            ('set_pixels', [N+0, N+1, N+2, color]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        return program
    
    @staticmethod
    def generateDrawCross(N, color):
        program = [
            ('div', [(N+0, '.width'), 2]),
            ('set_pixels', [N+0, N+1, (N+0, '.y'), color]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('div', [(N+0, '.height'), 2]),
            ('set_pixels', [N+0, (N+0, '.x'), N+1, color]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        return program

    def get_task_prog(self, bg_color):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        # Randomly select a color from 0-9, excluding bg_color
        possible_colors = [c for c in range(10) if c != bg_color]
        color = np.random.choice(possible_colors)

        if a < 0.25:
            options = ['top', 'bottom', 'left_col', 'right_col', 'center_vertical', 'center_horizontal']
            where = np.random.choice(options)
            return DrawingGenerator.generateDrawLine(N, where, color)
        elif a < 0.5:
            return DrawingGenerator.generateDrawContour(N, color)
        elif a < 0.75:
            return DrawingGenerator.generateDrawGrid(N, color)
        else:
            return DrawingGenerator.generateDrawCross(N, color)

    def generate(self, k=1):

        # step 1: generate input grid
        a = np.random.uniform()

        if a < 0.5:
            # Generate a random grid size between 3x3 and 30x30
            height = np.random.randint(3, 31)
            width = np.random.randint(3, 31)
            
            # Generate a random grid filled with values 0-9
            bg_color = np.random.randint(0, 10)
            input_grid = np.full((height, width), bg_color)
        else:
            input_grid = self.sampler.sample()
        
        input_grid_DSL = self.DSL.Grid(input_grid)
        initial_state = [input_grid_DSL]

        # step 2: prog ground truth
        # Count occurrences of each color in the input grid
        color_counts = {}
        for row in input_grid:
            for color in row:
                color_counts[color] = color_counts.get(color, 0) + 1
        
        # Find the most common color
        bg_color = max(color_counts.items(), key=lambda x: x[1])[0]

        prog = self.get_task_prog(bg_color)

        # step 3: use prog to generate target_grid
        token_seq_list = ProgUtils.convert_prog_to_token_seq(prog, self.DSL)
        output_grid_DSL = pi.execute(token_seq_list, initial_state, self.DSL)

        return input_grid, output_grid_DSL.cells, prog