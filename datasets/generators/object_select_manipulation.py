from ARC_gym.grid_sampling.grid_sampler import GridSampler
import AmotizedDSL.program_interpreter as pi
from AmotizedDSL.prog_utils import ProgUtils
import numpy as np
from datasets.generators.flip_generator import FlipGenerator

class ObjectSelectManipulationGenerator:

    def __init__(self, DSL):
        self.DSL = DSL
        self.first_call = True
        self.sampler = GridSampler()

    def _get_manip_prog(self, manip, N):
        if manip == 'flip':
            a = np.random.uniform()

            if a < 0.5:
                # h flip
                return FlipGenerator.generateHFlip(N, 1)
            else:
                # v flip
                return FlipGenerator.generateVFlip(N, 1)
            
    def get_task_prog(self, manip):
        N = len(self.DSL.semantics)

        manip_program = self._get_manip_prog(manip, N)

        # base program for selecting objects in the grid and applying the specified
        # manipulation to each of them.
        base_program = [
            ('get_objects', [N+0])
        ]

        base_program.extend(manip_program)

        # add the logic that overwrites the modified objects onto the original grid
        base_program.append(('rebuild_grid', [N+0, N+1]))
        base_program.append(('del', [N+0]))
        base_program.append(('del', [N+0]))

        return base_program

    def generate(self, k=1):

        manipulations = ['flip', 'rotate', 'filling', 'shearing', 'recoloring', 'completion',
                         'draw_grid', 'expansion', 'erosion', 'scaleup', 'scaledown']

        grid_categories_per_manip = {
            'flip': ['distinct_colors_adjacent'],
            'rotate': [], 
            'filling': [], 
            'shearing': [], 
            'recoloring': [], 
            'completion': [],
            'draw_grid': [], 
            'expansion': [], 
            'erosion': [], 
            'scaleup': [], 
            'scaledown': []
        }

        # Randomly pick an object manipulation to apply, and the grid generation categories
        # that are sensible for this manipulation.
        #manip = np.random.choice(manipulations)
        # TODO: temporary
        manip = 'flip'
        grid_categories = grid_categories_per_manip[manip]

        while True:
            # step 1: generate input grid
            input_grid, object_mask = self.sampler.sample_by_category(grid_categories)
            input_grid_DSL = self.DSL.Grid(input_grid)
            initial_state = [input_grid_DSL]

            # step 2: prog ground truth
            prog = self.get_task_prog(manip)

            print("==> Generated prog: ")
            for instr_step in prog:
                print(f"{instr_step}")

            # step 3: use prog to generate target_grid
            token_seq_list = ProgUtils.convert_prog_to_token_seq(prog[1:], self.DSL)

            # execution here is a bit tricky: we don't have a functional neural
            # get_objects since we need this data here to train it. Instead, we must
            # use directly the object_mask, generate the list of Grids from it, and
            # add it to the initial_state.
            # We skip the first instruction step because it's get_objects.
            grid_list = self.DSL.Grid.get_grid_list(input_grid, object_mask)

            initial_state.append(grid_list)
            output_grid_DSL = pi.execute(token_seq_list , initial_state, self.DSL)

            # Check if input and output grids are different
            if not np.array_equal(input_grid, output_grid_DSL.cells):
                return input_grid, output_grid_DSL.cells, object_mask, prog