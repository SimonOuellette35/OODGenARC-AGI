import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model.transformer_model import StandardTransformerModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import ARC_gym.utils.tokenization as tok
import ARC_gym.utils.visualization as viz
import AmotizedDSL.DSL as DSL
from utils.prog_utils import ProgUtils
from state_tokenizer import StateTokenizer
import search.program_interpreter as pi
import copy
import random

# ======================================================= Setup & Hyper-parameters =========================================================

# Set deterministic seed for reproducibility
DET_SEED = 555
torch.manual_seed(DET_SEED)
torch.cuda.manual_seed(DET_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(DET_SEED)
random.seed(DET_SEED)

DSL_size = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS

# Training Hyperparameters
num_epochs = 500
batch_size = 2
learning_rate = 0.0001
MAX_N = 5000
DATA_MAX_SEQ_LENGTH = 931  # there are two types of MAX_SEQ_LENGTHs.. the one in the training data, which can only contain grids up to 30x30
                           # and the one used by the StateTokenizer that can potentially deal with intermediate states that are grids larger
                           # than 30x30...

print("DSL_size = ", DSL_size)
model = StandardTransformerModel.instantiate_from_config_file('gridcoder_cfg.json')
device = "cuda"

# ========================================================= Loading Training Data =====================================================

def load_data(filename='training.json', max_samples=None):
    X_train = []
    programs = []

    try:
        with open(filename, 'r') as f:
            data_list = json.load(f)
            if max_samples is not None:
                data_list = data_list[:max_samples]

            for data in data_list:
                X_train.append(data['input_sequence'])
                programs.append(data['prog'])
                
    except json.JSONDecodeError:
        # Fallback for line-delimited JSON
        with open(filename, 'r') as f:
            # Read up to num_samples lines
            for i, line in enumerate(f):
                if max_samples is not None:
                    if i >= max_samples:
                        break
                    
                data = json.loads(line)
                X_train.append(data['input_sequence'])
                programs.append(data['prog'])

    # Convert to PyTorch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.long)
    programs = torch.tensor(np.array(programs), dtype=torch.long)
    return X_train, programs

# Load data with a progress bar
print("Loading training data...")
X_train, Y_train = load_data('training.json')
print("Loading validation data...")
X_val, Y_val = load_data('validation.json')

dataset_train = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# ========================================================= Model initialization & training =================================================

RESUME_MODEL = False
RESUME_TRAINING = False
state_tokenizer = StateTokenizer()

# Initialize model with dropout
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# Add learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, 
                             verbose=True, min_lr=1e-5)

if RESUME_MODEL:
    checkpoint = torch.load('gridcoder_intermediate.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

if RESUME_TRAINING:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Calculate and display the number of parameters in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_batch_programs(model, batch_output, target_grids, target_programs):
    '''
        * batch_output is: the last (or initial) variable state produced by the last step, in Python variable format.

        * tokenized_batch_output is: the last (or initial) variable state in tokenized form. It includes both the input and output initial
        grids, because it is used by the neural network.

        * batch_full_intermediate_state is: the entire list batch_output states for all instruction steps executed so far in the current program.
        Used by the execute_instruction_step call, since an instruction step to refer to any previously generated variable in the intermediate state.

        * target_programs is of shape: [batch_size, max instruction steps, max step sequence length]

        Returns predicted_sequences, of shape: [max instruction steps, batch_size, max step sequence length, target vocab]
    '''
    predicted_sequences = []
    tokenized_batch_output = []
    batch_full_intermediate_state = batch_output
    tokenized_targets = []
    for batch_idx in range(len(batch_output)):
        current_tokenized_states = []
        
        tokenized_target = state_tokenizer.tokenize(target_grids[batch_idx])
        tokenized_targets.append(tokenized_target)

        for state_idx in range(len(batch_output[batch_idx])):
            tmp = state_tokenizer.tokenize(batch_output[batch_idx][state_idx])
            if tmp is None:
                print("Couldn't tokenize: ", batch_output[batch_idx][state_idx])
                exit(-1)

            tmp = StateTokenizer.pad(tmp)
            current_tokenized_states.append(tmp)
           
        tokenized_batch_output.append(current_tokenized_states)

    # decoding idx is now offset by the initial number of intermediate states.
    max_instr_steps = max([len(program) for program in target_programs])

    # Pre-encode all intermediate states. Start the decoding step based on the number of previous states.
    with torch.no_grad():
        encoder_memory = model.get_encoder_memory(tokenized_batch_output, tokenized_targets)

    if encoder_memory is None:
        print("ERROR: pre_encode_states returned a None encoder_memory...")
        exit(-1)

    prog_step_idx = 0
    while prog_step_idx < max_instr_steps:

        # Generate the full instruction sequence for this program step
        tmp_pred_seq = model.decode(encoder_memory, target_programs[:, prog_step_idx], True)

        predicted_sequences.append(tmp_pred_seq)

        # Execute the instruction sequence to get the next state
        with torch.no_grad():
            tmp_batch_output = pi.execute_instruction_step_batch(target_programs[:, prog_step_idx], batch_full_intermediate_state, DSL)

            tmp_tokenized_batch_output = state_tokenizer.tokenize_batch(tmp_batch_output)

        for batch_idx in range(len(tmp_batch_output)):
            if isinstance(tmp_batch_output[batch_idx], pi.DeleteAction):
                batch_full_intermediate_state[batch_idx] = [
                    state for i, state in enumerate(batch_full_intermediate_state[batch_idx])
                    if i != tmp_batch_output[batch_idx].state_idx
                ]
            elif tmp_batch_output[batch_idx] is not None:
                batch_full_intermediate_state[batch_idx].append(tmp_batch_output[batch_idx])

        for batch_idx in range(len(tokenized_batch_output)):
            if tmp_tokenized_batch_output[batch_idx][0] == -1 and len(tmp_tokenized_batch_output[batch_idx]) == 1:
                # A DeleteAction occurred

                # Get the state index to delete from the DeleteAction
                state_idx_to_del = tmp_batch_output[batch_idx].state_idx
                # Filter out the element at index state_idx_to_del from tokenized_batch_output[batch_idx]
                if state_idx_to_del < len(tokenized_batch_output[batch_idx]):
                    tokenized_batch_output[batch_idx].pop(state_idx_to_del)
            elif tmp_tokenized_batch_output[batch_idx][0] == 0 and len(tmp_tokenized_batch_output[batch_idx]) == 1:
                # program complete, new state was None
                pass
            else:
                tokenized_batch_output[batch_idx].append(StateTokenizer.pad(tmp_tokenized_batch_output[batch_idx]))

        # must take tokenized batch output as input, not the raw variables...
        encoder_memory = model.get_encoder_memory(tokenized_batch_output, tokenized_targets)

        prog_step_idx += 1

    return predicted_sequences

total_params = count_parameters(model)
print(f"Model has {total_params:,} trainable parameters")

def test_program(programs, states, target_grids, start_grid_pos):
    for batch_idx in range(programs.shape[0]):
        # Apply prog to input and confirm that it generates output... validate the ground truth.
        # Split the input grid sequence at the first occurrence of 2
        intermediate_states = states[batch_idx]
        label_seq = programs[batch_idx].cpu().data.numpy()
        target_grid = target_grids[batch_idx]
        
        tmp_intermediate_states = copy.deepcopy(intermediate_states)
        pred_grid = pi.execute(label_seq, tmp_intermediate_states, DSL)

        if np.any(pred_grid.cells != target_grid.cells):
            print("==> ERROR: ground truth program not generating the expected outputs from given inputs.")
            
            # viz.draw_grid_triple(input_grid.cells, pred_grid.cells, target_grid.cells)

print(f"Training for {num_epochs} epochs on a total of {len(dataset_train)} samples...")
train_losses = []
# Training Loop

def create_initial_state(input_sequence, orig_programs, DSL):
    batch_output = []
    new_programs = []
    batch_start_state_pos = []
    targets = []

    for i in range(len(input_sequence)):
        detok_outp_grid = tok.detokenize_grid_unpadded(input_sequence[i][DATA_MAX_SEQ_LENGTH:].cpu().data.numpy())
        tmp_output_grid = DSL.Grid(detok_outp_grid)

        targets.append(tmp_output_grid)

        tmp_states = []
        # here, add the original grid
        detok_inp_grid = tok.detokenize_grid_unpadded(input_sequence[i][:DATA_MAX_SEQ_LENGTH].cpu().data.numpy())
        tmp_input_grid = DSL.Grid(detok_inp_grid)

        cur_state = tmp_input_grid
        tmp_states.append(cur_state)

        batch_output.append(tmp_states)
        batch_start_state_pos.append(0)
        
        new_prog = orig_programs[i]
        new_programs.append(torch.unsqueeze(new_prog, dim=0))

    return batch_output, targets, torch.cat(new_programs), batch_start_state_pos

model.train()
for epoch in range(num_epochs):

    total_loss = 0

    currentN = 0
    for grids_batch, progs_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", total=MAX_N-currentN):
        grids_batch, progs_batch = grids_batch.to(device), progs_batch.to(device)

        optimizer.zero_grad()

        batch_output, target_grids, renumbered_prog_gts, start_state_pos = create_initial_state(grids_batch, progs_batch, DSL)
        #test_program(renumbered_prog_gts, batch_output, target_grids, start_state_pos)
        predicted_sequences = generate_batch_programs(model, batch_output, target_grids, renumbered_prog_gts)

        # renumbered_prog_gts is of shape: [batch_size, max instruction steps, max step sequence length]
        # predicted_sequences is of shape: [max instruction steps, batch_size, max step sequence length, target vocab]

        loss = model.loss(predicted_sequences, renumbered_prog_gts)
        
        # Calculate accuracy by comparing predictions to target
        acc = 0.
        for decoding_idx, pred in enumerate(predicted_sequences):
            
            tgt = renumbered_prog_gts[:, decoding_idx, :]
            pred = torch.argmax(pred, dim=-1)
            matches = (pred == tgt[:, 1:]).float()
            acc += (matches.sum() / matches.numel()) * 100.0
            
        acc /= len(predicted_sequences)
        
        loss.backward()
        
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

        # Free memory
        del predicted_sequences, batch_output, target_grids, renumbered_prog_gts
        torch.cuda.empty_cache()
        
        currentN += 1

        if currentN >= MAX_N:
            break

    avg_loss = total_loss/len(train_loader)
    train_losses.append(avg_loss)
    
    # Update learning rate based on loss
    scheduler.step(avg_loss)
    
    # Save model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, 'gridcoder_intermediate.pth')
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.6f}, Train. acc.: {acc}")
