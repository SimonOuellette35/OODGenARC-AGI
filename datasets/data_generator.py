import numpy as np
import ARC_gym.utils.visualization as viz
import ARC_gym.utils.tokenization as tok
import AmotizedDSL.DSL as DSL
from AmotizedDSL.prog_utils import ProgUtils
from datasets.task_generator_factory import TaskGeneratorFactory
from tqdm import tqdm


tasks = {
    'shifts': None, 
    'rotations': None, 
    'flips': None, 
    'recoloring': None, 
    'flip+recoloring': None, 
    'shift+recoloring': None, 
    'flip+shift': None,
    'counting': None,
    'drawing': None,
    'shearing': None,
    'cropping': None
    }

def generate(model, num_samples=1000, k=1):

    # instantiate the task generators for all tasks! (because they are stateful...)
    for task_name, _ in tasks.items():
        tasks[task_name] = TaskGeneratorFactory.create(task_name, DSL)

    combined_input = []
    programs = []
    
    # Add progress bar
    for _ in tqdm(range(num_samples), desc="Generating samples"):
        # randomly select a task from the list
        task_name = np.random.choice(list(tasks.keys()))
        demonstration_set = []
        
        for _ in range(k):
            # use its generator to generate X, Y, prog for k = 1.
            task_generator = tasks[task_name]
            X, Y, prog = task_generator.generate(k=1)
    
            # Convert input/output grids to sequence with row markers
            tokenized_input_grid = tok.tokenize_grid(X)
            tokenized_output_grid = tok.tokenize_grid(Y)
                
            # Concatenate X and Y for the input
            grid_sequence = np.concatenate((tokenized_input_grid, tokenized_output_grid))
            demonstration_set.append(grid_sequence)
            
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

            if k == 1:
                combined_input.append(grid_sequence)
            else:
                demonstration_set.append(grid_sequence)

        # Ensure label_seq has exactly model.max_instr_steps entries
        if len(label_seq) < model.max_instr_steps:
            # Calculate how many additional sequences of max_instr_seq_length EOS_TOKEN we need
            num_additional_seqs = model.max_instr_steps - len(label_seq)
            # Create and add the additional sequences
            for _ in range(num_additional_seqs):
                eos_sequence = [ProgUtils.EOS_TOKEN] * model.max_target_seq_length
                label_seq.append(eos_sequence)

        if k > 1:
            combined_input.append(demonstration_set)
        programs.append(label_seq)

    return combined_input, programs
