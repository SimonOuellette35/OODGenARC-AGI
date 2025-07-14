import AmotizedDSL.DSL as DSL
import ARC_gym.utils.tokenization as tok
from torch.utils.data import Dataset
import numpy as np
from datasets.generators.arbitrary_task_generator import ArbitraryTaskGenerator
import torch


# This dataset uses the Hodel primitives, rather than the simplified ARC Gym primitives.
class ToyDataset(Dataset):
    def __init__(self, validation=False, ood = False, randomize_ordering=False):
        self.task_generator = ArbitraryTaskGenerator(validation)

        self.validation = validation
        self.randomize_ordering = randomize_ordering
        self.ood = ood

    def sample_transform(self):
        task_valid = False
        while not task_valid:
            task_valid = True
            try:
                return self.generate_random_task()
            except Exception as e:
                print("An error occurred:")
                import traceback
                traceback.print_exc()
                task_valid = False

    def __len__(self):
        return 1000

    def generateHFlip(self, N):
        program, ref = DSL.get_subroutine_hmirror(N+0, N+0)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def generateVFlip(self, N):
        program, ref = DSL.get_subroutine_vmirror(N+0, N+0)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def generateSetColor(self, N):
        program = [
            ('equal', [(N+0, '.c'), 0]),
            ('switch', [N+1, 0, 2]),
            ('del', [N+1]),
            ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]
        return program

    def generateTask1(self, N):
        hmirror_program, ref = DSL.get_subroutine_hmirror(N+0, N+0)

        hmirror_program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        hmirror_program.append(('del', [N+0]))
        hmirror_program.append(('del', [N+0]))

        set_fg_col_program = [
            ('equal', [(N+0, '.c'), 0]),
            ('switch', [N+1, 0, 2]),
            ('del', [N+1]),
            ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        program = hmirror_program
        program.extend(set_fg_col_program)

        return program

    def generateTask2(self, N):
        # program 1: vmirror, set fg color 3
        vmirror_program, ref = DSL.get_subroutine_vmirror(N+0, N+0)

        vmirror_program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        vmirror_program.append(('del', [N+0]))
        vmirror_program.append(('del', [N+0]))

        set_fg_col_program = [
            ('equal', [(N+0, '.c'), 0]),
            ('switch', [N+1, 0, 2]),
            ('del', [N+1]),
            ('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), N+1]),
            ('del', [N+0]),
            ('del', [N+0])
        ]

        program = vmirror_program
        program.extend(set_fg_col_program)

        return program

    def generateTask3(self, N):
        # program 3: shift pixels to the right
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

    def generateTask4(self, N):
        # right shift, then hflip
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

        hmirror_program, ref = DSL.get_subroutine_hmirror(N+0, N+0)
        hmirror_program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        hmirror_program.append(('del', [N+0]))
        hmirror_program.append(('del', [N+0]))

        program.extend(hmirror_program)
        return program

    def generateTask5(self, N):
        # vflip, right shift
        program, ref = DSL.get_subroutine_vmirror(N+0, N+0)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        # right shift:
        program.append(('add', [(N+0, '.x'), 1]))
        program.append(('set_pixels', [N+0, N+1, (N+0, '.y'), (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, 0, (N+1, '.y'), 0]))
        program.append(('del', [N+1]))
        program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def generateTask6(self, N):
        # hflip, shift up
        program, ref = DSL.get_subroutine_hmirror(N+0, N+0)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        # shift up:
        program.append(('sub', [(N+0, '.y'), 1]))
        program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
        program.append(('del', [N+1]))
        program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def generateTask7(self, N):
        # shift down, hflip
        program = []
        program.append(('add', [(N+0, '.y'), 1]))
        program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, (N+1, '.x'), 0, 0]))
        program.append(('del', [N+1]))
        program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        tmp_hflip, ref = DSL.get_subroutine_hmirror(N+0, N+0)
        program.extend(tmp_hflip)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def generateTask8(self, N):
        # shift up, vflip
        program = []
        program.append(('sub', [(N+0, '.y'), 1]))
        program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
        program.append(('del', [N+1]))
        program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        # vflip , shift up
        tmp_vflip, ref = DSL.get_subroutine_vmirror(N+0, N+0)
        program.extend(tmp_vflip)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def generateTask9(self, N):
        # vflip , shift up
        program, ref = DSL.get_subroutine_vmirror(N+0, N+0)
        program.append(('set_pixels', [N+0, (N+0, '.x'), (N+0, '.y'), ref]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        program.append(('sub', [(N+0, '.y'), 1]))
        program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
        program.append(('del', [N+1]))
        program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program

    def generateShiftUp(self, N):
        program = []
        program.append(('sub', [(N+0, '.y'), 1]))
        program.append(('set_pixels', [N+0, (N+0, '.x'), N+1, (N+0, '.c')]))
        program.append(('del', [N+1]))
        program.append(('set_pixels', [N+1, (N+1, '.x'), (N+1, '.max_y'), 0]))
        program.append(('del', [N+1]))
        program.append(('crop', [N+1, 0, 0, (N+0, '.width'), (N+0, '.height')]))
        program.append(('del', [N+0]))
        program.append(('del', [N+0]))

        return program
    
    def generateShiftDown(self, N):
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
    
    def not_determined(self, a, b, c):
        if a[0] == b[0] or a[0] == c[0] or b[0] == c[0]:
            return True
        else:
            return False

    def __getitem__(self, idx):
        S = {}

        task_valid = False
        while not task_valid:

            task_desc = -1
            if self.ood:
                program, task_desc = self.generateOODTasks()
            else:
                programs, task_desc = self.generateIIDTasks()

            # print("Generating data for program: ")
            # for row in program:
            #     print("==> ", row)

            combined_grids = self.generate_data(task_desc)

            example_grid_set1, _, label_seq1 = self.task_generator.generate(DSL, programs[0])
            example_grid_set2, _, label_seq2 = self.task_generator.generate(DSL, programs[1])
            example_grid_set3, _, label_seq3 = self.task_generator.generate(DSL, programs[2])

            if self.not_determined(example_grid_set1, example_grid_set2, example_grid_set3):
                continue

            if task_desc == 0:
                example_grid_set = example_grid_set1
                label_seq = label_seq1
            elif task_desc == 1:
                example_grid_set = example_grid_set2
                label_seq = label_seq2
            else:
                example_grid_set = example_grid_set3
                label_seq = label_seq3

            k = len(example_grid_set)

            # parse example_grid_set into x_batch and y_batch
            tmp_x_batch = [grid[0] for grid in example_grid_set]
            tmp_y_batch = [grid[1] for grid in example_grid_set]

            x_batch = []
            y_batch = []
            for example_idx in range(k):
                gridx = tmp_x_batch[example_idx]
                gridy = tmp_y_batch[example_idx]

                if len(gridx) > 6:
                    print("==> X Grid height > 6")
                    exit(0)
                if len(gridx[0]) > 6:
                    print("==> X Grid width > 6")
                    exit(0)
                if len(gridy) > 6:
                    print("==> Y Grid height > 6")
                    exit(0)
                if len(gridy[0]) > 6:
                    print("==> Y Grid width > 6")
                    exit(0)

                tmp_x = tok.tokenize_grid(tmp_x_batch[example_idx], max_length=MAX_SEQ_LENGTH)
                x_batch.append(tmp_x)
                
                tmp_y = tok.tokenize_grid(tmp_y_batch[example_idx], max_length=MAX_SEQ_LENGTH)
                y_batch.append(tmp_y)

            task_valid = True
            
        S['xs'] = x_batch
        S['ys'] = y_batch
        S['label_seq'] = label_seq
        S['task_desc'] = task_desc

        return S


