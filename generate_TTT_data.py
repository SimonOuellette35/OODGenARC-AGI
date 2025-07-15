import json
from tqdm import tqdm
from datasets.toy_data_generator import generate_data
from model.transformer_model import StandardTransformerModel
import ARC_gym.utils.tokenization as tok
import numpy as np


K = 10
trainN = 100000
valN = 250
model = StandardTransformerModel.instantiate_from_config_file('gridcoder_cfg.json')

# Generate training data
print(f"Generating {trainN} training samples...")

input_sequences, programs = generate_data(model, num_samples=trainN, k=K)

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

# Convert to list of dictionaries for JSON serialization
training_data = {}

for i in tqdm(range(trainN), desc="Generating training data"):
    key = f"{i:08d}"

    task_example = {}
    demonstration_set = []
    test_set = []
    for j in range(K):
        pair = {}

        input_seq = input_sequences[i][j]

        # split input_sequence into input_grid/target_grid
        grids = split_XY(input_seq)
        
        input_grid = tok.detokenize_grid_unpadded(grids[0])
        target_grid = tok.detokenize_grid_unpadded(grids[1])
    
        pair["input"] = input_grid
        pair["output"] = target_grid

        if j == K-1 or j == K-2:
            test_set.append(pair)
        else:
            demonstration_set.append(pair)

    task_example["train"] = demonstration_set
    task_example["test"] = test_set

    training_data[key] = task_example

def to_python_type(obj):
    """
    Recursively convert numpy types and tuples to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {to_python_type(k): to_python_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_type(i) for i in obj]
    elif isinstance(obj, tuple):
        return [to_python_type(i) for i in obj]  # Convert tuple to list
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    else:
        print(f"[DEBUG] Non-serializable type encountered: {type(obj)} with value: {obj}")
        return str(obj)  # fallback: convert to string

# Save training data to JSON file
with open("training_TTT.json", "w") as f:
    json.dump(to_python_type(training_data), f)

print("Training data saved to training_TTT.json")

# Generate validation data
print(f"Generating {valN} validation samples...")
input_sequences, programs = generate_data(model, num_samples=valN, k=K)

validation_data = {}

for i in range(valN):
    key = f"{i:08d}"

    task_example = {}
    demonstration_set = []
    test_set = []
    for j in range(K):
        pair = {}

        input_seq = input_sequences[i][j]

        # split input_sequence into input_grid/target_grid
        grids = split_XY(input_seq)
        
        input_grid = tok.detokenize_grid_unpadded(grids[0])
        target_grid = tok.detokenize_grid_unpadded(grids[1])
    
        pair["input"] = input_grid
        pair["output"] = target_grid

        if j == K-1 or j == K-2:
            test_set.append(pair)
        else:
            demonstration_set.append(pair)

    task_example["train"] = demonstration_set
    task_example["test"] = test_set

    validation_data[key] = task_example

# Save validation data to JSON file
with open("validation_TTT.json", "w") as f:
    json.dump(to_python_type(validation_data), f)

print("Validation data saved to validation_TTT.json")
