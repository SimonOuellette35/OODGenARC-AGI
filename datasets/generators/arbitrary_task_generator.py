import random
import numpy as np
from AmotizedDSL.prog_utils import ProgUtils
import AmotizedDSL.program_interpreter as pi
from typing import List, Tuple
import ARC_gym.utils.visualization as viz


class ArbitraryTaskGenerator:

    def __init__(self, DSL, validation=False):
        self.DSL = DSL
        self.validation = validation

    def generate_grids(self, k, dsl):
        grids = []
        for _ in range(k):
            height = 5
            width = 5
            
            # Initialize grid with 50% chance of color 0, otherwise random 1-9
            grid = np.zeros((height, width), dtype=int)
            mask = np.random.random((height, width)) > 0.5
            
            # Choose 2 random colors between 1 and 9
            colors = np.random.choice(np.arange(1, 10), size=2, replace=False)
            
            # Assign one of the two colors to the masked pixels
            grid[mask] = np.random.choice(colors, size=np.sum(mask))
            
            # Ensure there are at least 4 non-zero pixels
            non_zero_count = np.sum(grid > 0)
            if non_zero_count < 4:
                # Add more colored pixels if needed
                zero_indices = np.where(grid == 0)
                indices_to_color = np.random.choice(len(zero_indices[0]), size=4-non_zero_count, replace=False)
                for i in indices_to_color:
                    grid[zero_indices[0][i], zero_indices[1][i]] = np.random.choice(colors)
            
            # Convert to Grid object
            grids.append(dsl.Grid(grid))
            
        return grids

    # Applies specified program to 5x5 grids of randomized pixels
    def generate(self, dsl, program, k=1):
        N = len(dsl.semantics)

        task_invalid = True
        while task_invalid:
            try:
                input_grids = self.generate_grids(k, dsl)
                label_seq = ProgUtils.convert_prog_to_token_seq(program, dsl)

                example_grid_set = []
                for example_idx in range(len(input_grids)):

                    grid_inp = input_grids[example_idx]
                    grid_outp = pi.execute(label_seq, grid_inp, dsl)
                    
                    # viz.draw_grid_pair(grid_inp.cells, grid_outp.cells)
                    example_grid_set.append((grid_inp.cells, grid_outp.cells))

                    if grid_inp == grid_outp:
                        continue

                if len(example_grid_set) > 0:
                    task_invalid = False

            except KeyboardInterrupt:
                print("\nKeyboard interrupt detected. Stopping the program.")
                raise  # Re-raise the KeyboardInterrupt to stop the program
            except Exception as e:
                print(f"==> Task FAILED with an exception:")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Traceback:")
                import traceback
                traceback.print_exc()

        return example_grid_set, program, label_seq
    