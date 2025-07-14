import argparse
import torch
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_TIMEOUT = 180
VERBOSE = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="Command line arguments for GridCoder")

    parser.add_argument("--dataset", type=str, default="ood_data1.json", help="File to load")
    parser.add_argument("--time_budget", type=int, default=DEFAULT_TIMEOUT, help="Time budget per task in seconds")
    parser.add_argument("--skip", type=int, default=-1, help="Skip N tasks")
     
    args = parser.parse_args()
    return args

args = parse_arguments()

import AmotizedDSL.DSL as DSL
from datasets.toy_dataset import ToyDataset
from torch.utils.data import DataLoader
from model.transformer_model import StandardTransformerModel
import ARC_gym.utils.tokenization as tok
import search.tree_search as p_star
from AmotizedDSL.prog_utils import ProgUtils
from state_tokenizer import StateTokenizer

# ================================================================== Dataset ==================================================================
def load_data(num_samples=1000, filename='training.json'):
    X_train = []
    Y_train = []

    try:
        with open(filename, 'r') as f:
            data_list = json.load(f)
            # Take only up to num_samples
            data_list = data_list[:num_samples]
            
            for data in data_list:
                X_train.append(data['input_sequence'])
                Y_train.append(data['prog'])
                
    except json.JSONDecodeError:
        # Fallback for line-delimited JSON
        with open(filename, 'r') as f:
            # Read up to num_samples lines
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                    
                data = json.loads(line)
                X_train.append(data['input_sequence'])
                Y_train.append(data['prog'])
            
    # Convert to PyTorch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.long)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.long)
    
    return X_train, Y_train

ds = ToyDataset()
X_data, Y_data = load_data(100, args.dataset)
dataset_val = TensorDataset(X_data, Y_data)
eval_loader = DataLoader(dataset_val, batch_size=1)

# ================================================================== Model ==================================================================

# Set deterministic seed for reproducibility
DET_SEED = 123
torch.manual_seed(DET_SEED)
torch.cuda.manual_seed(DET_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(DET_SEED)

DSL_size = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS

# Hyperparameters
state_tokenizer = StateTokenizer()
model = StandardTransformerModel.instantiate_from_config_file('gridcoder_cfg.json')
device = "cuda"

model = model.to(device)
model.eval()

# ================================================================== Tasks ==================================================================

def display_program_sequence(program):
    # Start from the current node
    current = program
    sequence = []
    
    # Walk up the tree until we reach the root
    while current is not None:
        if current.parent_node is not None:
            # Get the instruction sequence at this step
            instruction = current.parent_node.instruction_seqs[current.instruction_idx]
            sequence.append(instruction)
        current = current.parent_node
    
    # Reverse the sequence since we built it from leaf to root
    sequence.reverse()
    
    # Display the sequence
    for i, instruction in enumerate(sequence):
        intermediate_seq = ProgUtils.convert_token_seq_to_token_tuple(instruction, DSL)
        handwritten_seq = ProgUtils.convert_token_tuple_to_str(intermediate_seq, DSL)
        print(f"Step {i}: {handwritten_seq}")

def process_task(model, X, Y):
    '''
    Parameters:
        @param X: list of integers representing the input grid, with 0, 1, 2 as special tokens, and 3 to 12 being the colors.
        @param Y: list of integers representing the input grid, with 0, 1, 2 as special tokens, and 3 to 12 being the colors.

    Returns:
        success: True or False, if the solution was found.
        program: Description of the program that solves the task.
    '''
    max_iterations = 1000000
    #try:
    input_grid = DSL.Grid(tok.detokenize_grid_unpadded(X))
    target_grid = DSL.Grid(tok.detokenize_grid_unpadded(Y))
    success, program = p_star.search(model, input_grid, target_grid, args.time_budget, max_iterations, verbose=VERBOSE)
    if success:
        print("Success! Program found: ", program)
    # except:
    #     import traceback
    #     print("Exception occurred during search:")
    #     traceback.print_exc()
    #     return False, None

    display_program_sequence(program)    
    
    return success, program


for task_idx, eval_task in enumerate(eval_loader):

    print("Task description/class ID: ", eval_task[1].cpu().data.numpy()[0][1])

    def split_XY(sequence):
        """
        Split a sequence into sub-lists, where each sub-list is terminated by the occurrence of integer 2.
        
        Args:
            sequence: A list of integers
            
        Returns:
            A list of sub-lists, where each sub-list ends with the integer 2
        """
        result = []
        current_sublist = []
        
        for num in sequence:
            current_sublist.append(num)
            if num == 2:
                result.append(current_sublist)
                current_sublist = []
        
        # Handle case where the sequence doesn't end with 2
        if current_sublist:
            result.append(current_sublist)
            
        return result

    if task_idx < args.skip:
        continue

    grids = split_XY(eval_task[0][0].cpu().data.numpy())
    
    process_task(model, grids[0], grids[1])

    # TODO: collect and display success rate
