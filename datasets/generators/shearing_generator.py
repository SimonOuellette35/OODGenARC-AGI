from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np


class ShearingGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    @staticmethod
    def generateShearRight(N):
        # 3cd86f4f
        # 423a55dc, but to the right
        program = [
            ('sub', [(N+0, '.height'), (N+0, '.y')]),       # These are the offsets to apply at each value of y

            # Shift each row to the right by its associated offset (N+1)
            ('add', [(N+0, '.x'), N+1]),                    # New starting x values for each row
            ('del', [N+1]),

            ('new_grid', [(N+0, '.width'), (N+0, '.height'), 0]),
            ('set_pixels', [N+2, N+1, (N+0, '.y'), (N+0, '.c')]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    @staticmethod
    def generateShearLeft(N):
        # 3cd86f4f, but to the left
        # 423a55dc
        program = [
            # Shift each row to the right by its associated offset (N+1)
            ('add', [(N+0, '.x'), (N+0, '.y')]),                    # New starting x values for each row
            ('new_grid', [(N+0, '.width'), (N+0, '.height'), 0]),
            ('set_pixels', [N+2, N+1, (N+0, '.y'), (N+0, '.c')]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    @staticmethod
    def generateShearZigZagBottomUp(N):
        # 1c56ad9f
        program = [
            # Get cyclical offsets to apply at each row
            ('sub', [(N+0, '.max_y'), (N+0, '.y')]),
            ('sin_half_pi', [N+1]),
            ('sub', [0, N+2]),
            ('del', [N+1]),
            ('del', [N+1]),

            # At this stage: N+0 = input grid, N+1 = cyclical offsets to apply per row
            # Shift each row to the right by its associated offset (N+1)
            ('add', [(N+0, '.x'), N+1]),                    # New starting x values for each row
            ('del', [N+1]),
            ('add', [N+1, 1]),                              # Offsets start at 0, not -1
            ('del', [N+1]),

            # At this stage: N+0 = input grid, N+1 = x offsets to paste to
            ('add', [(N+0, '.width'), 1]),                  # New grid is wider to make room for the left column
            ('new_grid', [N+2, (N+0, '.height'), 0]),
            ('del', [N+2]),

            # At this stage: N+0 = input grid, N+1 = x offsets to paste to, N+2 = new grid
            ('set_pixels', [N+2, N+1, (N+0, '.y'), (N+0, '.c')]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])      # THIS IS AN OVERDELETE??? Not really, in theory. So what is going on?
        ]
        return program

    @staticmethod
    def generateShearZigZagBottomUp2(N):
        program = [
            # Get cyclical offsets to apply at each row
            ('sub', [(N+0, '.max_y'), (N+0, '.y')]),
            ('cos_half_pi', [N+1]),
            ('sub', [0, N+2]),
            ('del', [N+1]),
            ('del', [N+1]),
            
            # At this stage: N+0 = input grid, N+1 = cyclical offsets to apply per row
            # Shift each row to the right by its associated offset (N+1)
            ('add', [(N+0, '.x'), N+1]),                    # New starting x values for each row
            ('del', [N+1]),
            ('add', [N+1, 1]),                              # Offsets start at 0, not -1
            ('del', [N+1]),

            # At this stage: N+0 = input grid, N+1 = x offsets to paste to
            ('add', [(N+0, '.width'), 1]),                  # New grid is wider to make room for the left column
            ('new_grid', [N+2, (N+0, '.height'), 0]),
            ('del', [N+2]),

            # At this stage: N+0 = input grid, N+1 = x offsets to paste to, N+2 = new grid
            ('set_pixels', [N+2, N+1, (N+0, '.y'), (N+0, '.c')]),
            ('del', [N+0]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    def get_task_prog(self):
        a = np.random.uniform()

        N = len(self.DSL.semantics)

        if a < 0.25:
            return ShearingGenerator.generateShearRight(N)
        elif a < 0.5:
            return ShearingGenerator.generateShearLeft(N)
        elif a < 0.75:
            return ShearingGenerator.generateShearZigZagBottomUp(N)
        else:
            return ShearingGenerator.generateShearZigZagBottomUp2(N)

    def generate(self, k=1):

        while True:
            # step 1: generate input grid
            min_dim = 6
            max_dim = 20
            width = np.random.randint(min_dim, max_dim + 1)
            height = np.random.randint(min_dim, max_dim + 1)
            bg_color = 0
            input_grid = [[bg_color for _ in range(width)] for _ in range(height)]

            # Select type of generation
            gen_type = np.random.choice(['full', 'randomized', 'hollow'])
            
            # Select a section that's at least 5 pixels high
            min_height = 5
            max_height = height
            section_height = np.random.randint(min_height, max_height + 1)
            start_y = np.random.randint(0, height - section_height + 1)
            
            # Select width of section
            min_width = 3
            max_width = width
            section_width = np.random.randint(min_width, max_width + 1)
            start_x = np.random.randint(0, width - section_width + 1)
            
            # Select color (not background)
            color = np.random.choice([c for c in range(10) if c != bg_color])
            
            if gen_type == 'full':
                # Fill entire section with same color
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        input_grid[y][x] = color
                        
            elif gen_type == 'randomized':
                # Fill section with random colors
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        input_grid[y][x] = np.random.choice([c for c in range(10) if c != bg_color])
                        
            else:  # hollow
                # Create outline only
                for y in range(start_y, start_y + section_height):
                    for x in range(start_x, start_x + section_width):
                        if (y == start_y or y == start_y + section_height - 1 or 
                            x == start_x or x == start_x + section_width - 1):
                            input_grid[y][x] = color
                        else:
                            input_grid[y][x] = bg_color

            # Check if input grid has at least 10 non-zero pixels
            cells_int = np.array([list(row) for row in input_grid]).astype(int)
            if np.sum(cells_int > 0) < 10:
                continue
                
            input_grid_DSL = self.DSL.Grid(input_grid)
            initial_state = [input_grid_DSL]

            # step 2: prog ground truth
            prog = self.get_task_prog()

            # step 3: use prog to generate target_grid
            token_seq_list = ProgUtils.convert_prog_to_token_seq(prog, self.DSL)
            output_grid_DSL = pi.execute(token_seq_list, initial_state, self.DSL)

            # Check if input and output grids are different and output is not all black
            cells_int = np.array([list(row) for row in output_grid_DSL.cells]).astype(int)
            if not np.array_equal(input_grid, output_grid_DSL.cells) and not np.all(cells_int == 0):
                # Check if output grid dimensions are within limits
                output_height = len(output_grid_DSL.cells)
                output_width = len(output_grid_DSL.cells[0])
                if output_height > 30 or output_width > 30:
                    continue
                return input_grid, output_grid_DSL.cells, prog