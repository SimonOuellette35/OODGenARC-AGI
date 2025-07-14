import torch
import numpy as np
import ARC_gym.utils.visualization as viz
import ARC_gym.utils.tokenization as tok
import AmotizedDSL.DSL as DSL
from AmotizedDSL.prog_utils import ProgUtils
from datasets.toy_dataset import ToyDataset
from ARC_gym.grid_sampling.grid_sampler import GridSampler


ds = ToyDataset()


def get_gridcoderv1_task(grid, task_class_idx):
    hflip_grid = np.flip(grid, axis=1)  # Flip horizontally by flipping columns   
    vflip_grid = np.flip(grid, axis=0)  # Flip horizontally by flipping columns   

    if np.all(hflip_grid == vflip_grid):
        # Under-determined grid... don't keep it.
        return None, None

    prog = None
    if task_class_idx == 0:
        result_grid = hflip_grid
        prog = ['hmirror', 'EOS']
    elif task_class_idx == 1:
        result_grid = vflip_grid
        prog = ['vmirror', 'EOS']
    elif task_class_idx == 2:
        # Set foreground color to 2
        result_grid = np.where(grid > 0, np.array(2, dtype=np.int64), grid)
        prog = ['set_fg_color2', 'EOS']
    elif task_class_idx == 3:
        # Right shift
        result_grid = np.zeros_like(grid)
        result_grid[:, 1:] = grid[:, :-1]               # Shift all other columns right
        prog = ['shift_right', 'EOS']
    elif task_class_idx == 4:
        # hflip, set_fg_color
        result_grid = np.where(hflip_grid > 0, np.array(2, dtype=np.int64), hflip_grid)
        prog = ['hmirror', 'NEW_LEVEL', 'set_fg_color2', 'EOS']
    elif task_class_idx == 5:
        # vflip, set_fg_color
        result_grid = np.where(vflip_grid > 0, np.array(2, dtype=np.int64), vflip_grid)
        prog = ['vmirror', 'NEW_LEVEL', 'set_fg_color2', 'EOS']
    elif task_class_idx == 6:
        # right_shift, hflip
        rshift = np.zeros_like(grid)
        rshift[:, 1:] = grid[:, :-1]                    # Shift all other columns right
        result_grid = np.flip(rshift, axis=1)
        prog = ['shift_right', 'NEW_LEVEL', 'hmirror', 'EOS']
    elif task_class_idx == 7:
        # vflip, right_shift
        result_grid = np.zeros_like(vflip_grid)
        result_grid[:, 1:] = vflip_grid[:, :-1]         # Shift all other columns right
        prog = ['vmirror', 'NEW_LEVEL', 'shift_right', 'EOS']
    elif task_class_idx == 8:
        # hflip, shift up
        result_grid = np.zeros_like(hflip_grid)
        result_grid[:-1, :] = hflip_grid[1:, :]
        prog = ['hmirror', 'NEW_LEVEL', 'set_fg_color2', 'EOS']
    elif task_class_idx == 9:
        # shift down, hflip
        result_grid = np.zeros_like(hflip_grid)
        result_grid[1:, :] = hflip_grid[:-1, :]
        prog = ['shift_down', 'NEW_LEVEL', 'hmirror', 'EOS']
    elif task_class_idx == 10:
        # shift up, vflip
        ushift = np.zeros_like(grid)
        ushift[:-1, :] = grid[1:, :]
        result_grid = np.flip(ushift, axis=0)
        prog = ['shift_up', 'NEW_LEVEL', 'vmirror', 'EOS']
    elif task_class_idx == 11:
        # vflip, shift up
        result_grid = np.zeros_like(vflip_grid)
        result_grid[:-1, :] = vflip_grid[1:, :]
        prog = ['vmirror', 'NEW_LEVEL', 'shift_up', 'EOS']
    elif task_class_idx == 12:
        # shift up only
        result_grid = np.zeros_like(grid)
        result_grid[:-1, :] = grid[1:, :]
        prog = prog = ['shift_up', 'EOS']
    elif task_class_idx == 13:
        # shift down only
        result_grid = np.zeros_like(grid)
        result_grid[1:, :] = grid[:-1, :]
        prog = ['shift_down', 'EOS']
    else:
        print(f"==> ERROR: unknown task class {task_class_idx}")
        exit(0)

    return result_grid, prog


def get_training_task(grid, task_class_idx, DSL_size):
    hflip_grid = np.flip(grid, axis=1)  # Flip horizontally by flipping columns   
    vflip_grid = np.flip(grid, axis=0)  # Flip horizontally by flipping columns   

    if np.all(hflip_grid == vflip_grid):
        # Under-determined grid... don't keep it.
        return None, None

    prog = None
    if task_class_idx == 0:
        result_grid = hflip_grid
        prog = ds.generateHFlip(DSL_size)
    elif task_class_idx == 1:
        result_grid = vflip_grid
        prog = ds.generateVFlip(DSL_size)
    elif task_class_idx == 2:
        # Set foreground color to 2
        result_grid = np.where(grid > 0, np.array(2, dtype=np.int64), grid)
        prog = ds.generateSetColor(DSL_size)
    elif task_class_idx == 3:
        # Right shift
        result_grid = np.zeros_like(grid)
        result_grid[:, 1:] = grid[:, :-1]               # Shift all other columns right
        prog = ds.generateTask3(DSL_size)
    elif task_class_idx == 4:
        # hflip, set_fg_color
        result_grid = np.where(hflip_grid > 0, np.array(2, dtype=np.int64), hflip_grid)
        prog = ds.generateTask1(DSL_size)
    elif task_class_idx == 5:
        # vflip, set_fg_color
        result_grid = np.where(vflip_grid > 0, np.array(2, dtype=np.int64), vflip_grid)
        prog = ds.generateTask2(DSL_size)
    elif task_class_idx == 6:
        # right_shift, hflip
        rshift = np.zeros_like(grid)
        rshift[:, 1:] = grid[:, :-1]                    # Shift all other columns right
        result_grid = np.flip(rshift, axis=1)
        prog = ds.generateTask4(DSL_size)
    elif task_class_idx == 7:
        # vflip, right_shift
        result_grid = np.zeros_like(vflip_grid)
        result_grid[:, 1:] = vflip_grid[:, :-1]         # Shift all other columns right
        prog = ds.generateTask5(DSL_size)
    elif task_class_idx == 8:
        # hflip, shift up
        result_grid = np.zeros_like(hflip_grid)
        result_grid[:-1, :] = hflip_grid[1:, :]
        prog = ds.generateTask6(DSL_size)
    elif task_class_idx == 9:
        # shift down, hflip
        result_grid = np.zeros_like(hflip_grid)
        result_grid[1:, :] = hflip_grid[:-1, :]
        prog = ds.generateTask7(DSL_size)
    elif task_class_idx == 10:
        # shift up, vflip
        ushift = np.zeros_like(grid)
        ushift[:-1, :] = grid[1:, :]
        result_grid = np.flip(ushift, axis=0)
        prog = ds.generateTask8(DSL_size)
    elif task_class_idx == 11:
        # vflip, shift up
        result_grid = np.zeros_like(vflip_grid)
        result_grid[:-1, :] = vflip_grid[1:, :]
        prog = ds.generateTask9(DSL_size)
    elif task_class_idx == 12:
        # shift up only
        result_grid = np.zeros_like(grid)
        result_grid[:-1, :] = grid[1:, :]
        prog = ds.generateShiftUp(DSL_size)
    elif task_class_idx == 13:
        # shift down only
        result_grid = np.zeros_like(grid)
        result_grid[1:, :] = grid[:-1, :]
        prog = ds.generateShiftDown(DSL_size)
    else:
        print(f"==> ERROR: unknown task class {task_class_idx}")
        exit(0)

    return result_grid, prog

def get_ood_task(grid, task_class_idx):
    hflip_grid = np.flip(grid, axis=1)  # Flip horizontally by flipping columns

    # Create right-shifted grid
    rshift_grid = np.zeros_like(grid)
    rshift_grid[:, 1:] = grid[:, :-1]  # Shift all other columns right

    if task_class_idx == 0:
        # Convert all non-zero values in the shifted grids to 2
        result_grid = np.where(rshift_grid > 0, 2, rshift_grid)
    elif task_class_idx == 1:
        result_grid = np.zeros_like(grid)
        result_grid[:, 1:] = hflip_grid[:, :-1]  # Shift all other columns right
    elif task_class_idx == 2:
        # shift up and shift right
        result_grid = np.zeros_like(grid)
        result_grid[:-1, :] = rshift_grid[1:, :]
    elif task_class_idx == 3:
        # hflip + vflip = rot180
        result_grid = np.flip(hflip_grid, axis=0)
    elif task_class_idx == 4:
        # hflip, shift right, vflip
        result_grid = np.zeros_like(grid)
        result_grid[:, 1:] = hflip_grid[:, :-1]
        result_grid = np.flip(result_grid, axis=0)
    elif task_class_idx == 5:
        # set_fg_color, shift right, shift up
        result_grid = np.where(rshift_grid > 0, 2, rshift_grid)

        # Clone the array to avoid in-place modification issues
        temp_grid = result_grid.copy()
        result_grid[:-1, :] = temp_grid[1:, :]
        result_grid[-1, :] = 0
    elif task_class_idx == 6:
        # left shift
        result_grid = np.zeros_like(grid)
        result_grid[:, :-1] = grid[:, 1:]
    else:
        print("Invalid OOD task: ", task_class_idx)
        exit(0)

    return result_grid, []

def generate_data(model, ood=False, specified_task=None, num_samples=1000, k=1):
    if ood:
        NUM_TASKS = 7
    else:
        NUM_TASKS = 14
    DSL_size = len(DSL.semantics)
    programs = []
    sampler = GridSampler()

    combined_input = []

    # Generate data for each sample
    i = 0
    while i < num_samples:
        if specified_task is None:
            # Randomly choose between horizontal and vertical flip
            task_class_idx = np.random.choice(np.arange(NUM_TASKS))
        else:
            task_class_idx = specified_task

        task_input_sequences = []
        j = 0
        while j < k:
            # These tasks use the set_fg_color primitive, so don't use a monochrome grid as starting point
            if ood:
                if task_class_idx in [0, 5]:
                    grid = sampler.sample(monochrome_grid_ok=False, bg_color=0)
                else:
                    grid = sampler.sample()
            else:
                if task_class_idx in [2, 4, 5]:
                    grid = sampler.sample(monochrome_grid_ok=False, bg_color=0)
                else:
                    grid = sampler.sample()

            # Convert input grid to sequence with row markers
            tokenized_input_grid = tok.tokenize_grid(grid)

            if ood:
                result_grid, prog = get_ood_task(grid, task_class_idx)
            else:
                result_grid, prog = get_training_task(grid, task_class_idx, DSL_size)
                #result_grid, prog = get_gridcoderv1_task(grid, task_class_idx)
                
            if result_grid is None:
                continue

            # Convert flipped grid to sequence with row markers
            # print("Task IDX: ", task_class_idx)
            # viz.draw_grid_pair(grid, result_grid)
            tokenized_output_grid = tok.tokenize_grid(result_grid)
            
            if np.array_equal(tokenized_input_grid, tokenized_output_grid):
                continue

            # Concatenate X and Y for the input
            grid_sequence = np.concatenate((tokenized_input_grid, tokenized_output_grid))

            if k == 1:
                combined_input.append(list(grid_sequence))
            else:
                task_input_sequences.append(grid_sequence)

            j += 1

        if k > 1:
            combined_input.append(task_input_sequences)

        # Create label with SOS token
        label_seq = ProgUtils.convert_prog_to_token_seq(prog, DSL)

        # Pad the sequence up to max_instr_seq_length with EOS_token
        for idx, instr_step in enumerate(label_seq):
            if len(instr_step) < model.max_target_seq_length:
                padding = [ProgUtils.EOS_TOKEN] * (model.max_target_seq_length - len(instr_step))
                label_seq[idx] = instr_step + padding
            elif len(label_seq) > model.max_target_seq_length:
                print("==> ERROR: the training ground truth instruction step is longer than max_instr_seq_length!")
                exit(-1)

        # Ensure label_seq has exactly model.max_instr_steps entries
        if len(label_seq) < model.max_instr_steps:
            # Calculate how many additional sequences of max_instr_seq_length EOS_TOKEN we need
            num_additional_seqs = model.max_instr_steps - len(label_seq)
            # Create and add the additional sequences
            for _ in range(num_additional_seqs):
                eos_sequence = [ProgUtils.EOS_TOKEN] * model.max_target_seq_length
                label_seq.append(eos_sequence)

        programs.append(label_seq)

        i += 1

    return combined_input, programs

