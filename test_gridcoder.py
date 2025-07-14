import argparse
import os
import sys
import torch
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_TIMEOUT = 180
VERBOSE = True

def parse_arguments():
    parser = argparse.ArgumentParser(description="Command line arguments for GridCoder")

    parser.add_argument("--task", type=str, help="Task to load for eval or training ARC set: the name of the filename for the task")
    parser.add_argument("--dataset", type=str, default="eval", help="Task to load ('synthetic' for synthetically generated tasks, 'eval' for ARC evaluation dataset, 'train' for ARC training dataset)")
    parser.add_argument("--time_budget", type=int, default=DEFAULT_TIMEOUT, help="Time budget per task in seconds")
    parser.add_argument("--skip", type=int, default=-1, help="Skip N tasks")
     
    args = parser.parse_args()
    return args

args = parse_arguments()

if args.task == 'Kaggle':
    os.chdir('/kaggle/working/GridCoder/')
    sys.path.append('/kaggle/working/GridCoder/')
    print("==> Current working directory: ", os.getcwd())

import AmotizedDSL.DSL as DSL
from datasets.toy_dataset import ToyDataset
from torch.utils.data import DataLoader
from model.transformer_model import StandardTransformerModel
import ARC_gym.utils.tokenization as tok
import search.tree_search as p_star
#import search.tree_search_MCD as p_star
import utils.grid_utils as g
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

if args.task == 'Kaggle':
    # Load and parse the JSON file
    with open('/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json', 'r') as f:
    #with open('/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json', 'r') as f:
        data = json.load(f)
    
        # Store each task object in a list
        test_tasks = []
        for task_id, task_data in data.items():
            test_tasks.append({task_id: task_data})

        print("Loaded %i test tasks!" % len(test_tasks))

if args.task == 'alpha_POC':
    ds = ToyDataset()
    X_data, Y_data = load_data(100, args.dataset)
    dataset_val = TensorDataset(X_data, Y_data)
    eval_loader = DataLoader(dataset_val, batch_size=1)

# TODO: update this
# if args.dataset == 'eval':
#     print("Testing on ARC eval dataset.")
#     ds = ARCEvaluationDataset()
#     eval_loader = DataLoader(ds,
#                             batch_size=1,
#                             collate_fn=lambda x: make_gridcoder_batch(x),
#                             shuffle=False)
    

# TODO: implement ARC training dataset as well

# ================================================================== Heuristic ==================================================================

print("Evaluating learned probability heuristic")

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

if args.task == 'Kaggle':
    checkpoint = torch.load('/kaggle/working/GridCoder/gridcoder_intermediate.pth')
else:
    checkpoint = torch.load('gridcoder_intermediate.pth')

model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# ================================================================== Tasks ==================================================================

device = 'cuda'

def save_submissions(submissions):
    # Convert submissions dictionary to JSON-serializable format
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        else:
            return obj

    json_safe_submissions = convert_to_json_serializable(submissions)
    
    # Save submissions dictionary to JSON file
    with open('/kaggle/working/submission.json', 'w') as f:
        json.dump(json_safe_submissions, f)

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


def preprocess_kaggle_data(examples):

    def process_grid_pair(input_grid, output_grid):
        # Convert input_grid and output_grid lists into tuples of tuples
        cells_x = tuple(tuple(row) for row in input_grid)
        cells_y = tuple(tuple(row) for row in output_grid)

        support_x = tok.tokenize_grid(cells_x, max_length=931)
        support_y = tok.tokenize_grid(cells_y, max_length=931)

        GRID_LENGTH = (31 * 30) + 1              # 931
        x_token_seq = support_x[:GRID_LENGTH]
        y_token_seq = support_y[:GRID_LENGTH]

        x_tensor = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(x_token_seq), model.input_vocab_size)).to(device).float(), dim=0)
        y_tensor = torch.unsqueeze(torch.from_numpy(g.one_hot_encode(g.gridify(y_token_seq), model.input_vocab_size)).to(device).float(), dim=0)

        return x_tensor, x_token_seq, y_tensor, y_token_seq
    
    X_tensors = []
    Y_tensors = []
    X_token_seqs = []
    Y_token_seqs = []

    for example in examples:
        input_grid = example['input']

        if 'output' in example:
            output_grid = example['output']
        else:
            output_grid = input_grid

        x_tensor, x_token_seq, y_tensor, y_token_seq = process_grid_pair(input_grid, output_grid)

        X_tensors.append(x_tensor)
        X_token_seqs.append(x_token_seq)
        Y_tensors.append(y_tensor)
        Y_token_seqs.append(y_token_seq)

    return X_tensors, Y_tensors, X_token_seqs, Y_token_seqs

if args.task == 'Kaggle':

    # Expected submission json format:
    # {"00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    #  "009d5c81": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    #  "12997ef3": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
    #               {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
    #  ...
    # }
    submissions = {}
    print("Found %i test tasks!" % len(test_tasks))
    task_counter = 0
    for test_task in test_tasks:
        key = list(test_task.keys())[0]
        task_counter += 1
        #print(key)
        #if key == '32e9702f':
        print("Processing task #%i: %s" % (task_counter,  key))
        train_examples = test_task[key]['train']

        X_tensor, Y_tensor, X_token_seq, Y_token_seq = preprocess_kaggle_data(train_examples)
        
        solution, c1, c2, success = process_task(model, X_tensor, Y_tensor, X_token_seq, Y_token_seq)

        if success:
            print("==> Success ! Solution: ", solution)
        else:
            print("Failed to find a solution.")

        test_examples = test_task[key]['test']
        _, _, X_token_seq, Y_token_seq = preprocess_kaggle_data(test_examples)

        submission_list = []
        print("There are %i test examples!" % (len(test_examples)))

        for test_idx in range(len(test_examples)):

            if solution is None:
                result = X_token_seq[test_idx]

                result_grid = tok.detokenize_grid_unpadded(result)
                # Convert tuple of tuples to list of lists
                result_list = []
                for row in result_grid:
                    result_list.append(list(row))
            else:
                try:
                    # handle color_change case
                    if c1 is not None:
                        result, _, _ = p_star.get_prediction(solution, [X_token_seq[test_idx]], c1, c2)
                    else:
                        result, _, _ = p_star.get_prediction(solution, [X_token_seq[test_idx]])

                    # Convert tuple of tuples to list of lists
                    result_list = []
                    for row in result[0].cells:
                        result_list.append(list(row))
                except:
                    print("Exception occurred during prediction:")
                    import traceback
                    traceback.print_exc()
                    result = X_token_seq[test_idx]

                    result_grid = tok.detokenize_grid_unpadded(result)
                    # Convert tuple of tuples to list of lists
                    result_list = []
                    for row in result_grid:
                        result_list.append(list(row))

            attempt_json = {
                'attempt_1': result_list,
                'attempt_2': result_list
            }
            submission_list.append(attempt_json)

        print("Adding key %s to submissions." % key)
        submissions[key] = submission_list

        save_submissions(submissions)

else:

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

        # TODO: temporary, to simplify debugging.
        exit(0)