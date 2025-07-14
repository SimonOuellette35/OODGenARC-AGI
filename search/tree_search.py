import math
import numpy as np
import torch
import time
from search.search_tree_node import SearchTreeNode
from state_tokenizer import StateTokenizer
import AmotizedDSL.DSL as DSL
from AmotizedDSL.prog_utils import ProgUtils
import AmotizedDSL.program_interpreter as pi
import search.tree_search_common as ts

# Suppress scientific notation in numpy arrays
np.set_printoptions(suppress=True)
DET_SEED = 12345
np.random.seed(DET_SEED)
DSL_size = len(DSL.semantics) + ProgUtils.NUM_SPECIAL_TOKENS
MEMORY_LIMIT = 5        # was: 5
VERBOSE = False
VIZ = False
MAX_TREE_DEPTH = 20     # was: 20
PRIORITIZATION = 'joint_prob' # can be 'joint_prob', 'average_prob'
THRESH = 0.01           # was: 0.01

ENTROPY_INC = 0.01  # was: 0.1
ENTROPY_PROB_INC = 0.02 # was: 0.02

state_tokenizer = StateTokenizer()

# This contains the conviction of the best unvisited node at each decoding step.
global_leaf_stats = []


def enumerate_programs(model, selected_node, target_memory, threshold=THRESH, entropy=0.0, device='cuda'):
    '''
    Enumerates possible program sequences by recursively selecting tokens based on their probabilities.
    
    Parameters:
        @param model: the PyTorch probability model.
        @param selected_node: a SearchTreeNode instance representing where we are in the search tree.
        @param target_memory: already encoded with state_idx, target grids to reach.
        @param verbose: whether to print verbose output.
        @param thresh: threshold below which token probabilities are ignored.
        @param max_num_sequences: maximum number of sequences to generate.
        
    Returns:
        Updates the selected_node with instruction sequences, uncertainties, and log probabilities.
    '''

    initial_sequence = [0]   
    complete_sequences = []

    # These probabilities are the joint-probabilities of the individual instruction step sequences,
    # not the full cumulative probability of the whole program up to this point.
    sequence_probabilities = []
    
    with torch.no_grad():
        encoder_memory, memory_len = ts.prepare_encoder_memory(model, selected_node, target_memory, state_tokenizer, DSL_size, device)
        
    def valid_partial_sequence(current_sequence):
        if len(current_sequence) < 1:
            return True
        
        if current_sequence[0] != 0:
            return False

        if len(current_sequence) < 2:
            return True
        
        if current_sequence[1] < (10 + ProgUtils.NUM_SPECIAL_TOKENS) or current_sequence[1] > (25 + ProgUtils.NUM_SPECIAL_TOKENS):    # Note: this changes with the DSL
            return False

        if len(current_sequence) < 3:
            return True

        if current_sequence[2] != 1:
            return False

        return True

    def recurse(current_sequence, current_prob=1.0, depth=0):
        if not valid_partial_sequence(current_sequence):
            return
        
        # Base case: max depth reached or last token is EOS
        if depth >= model.max_target_seq_length - 1:
            return
        
        if current_sequence and len(current_sequence) > 1 and current_sequence[-1] == ProgUtils.EOS_TOKEN:
            complete_sequences.append(current_sequence)
            sequence_probabilities.append(current_prob)
            return
        
        # Get probability distribution for next token
        with torch.no_grad():
            _, probs = model.generate(encoder_memory, ProgUtils.SOS_TOKEN, init_sequence=current_sequence, iter_max=1)
        
        probs = probs[0].cpu().data.numpy()    # probs for only 1 token ahead.
        probs = probs[0]    # batch size is 1.
        if depth == 0:
            for idx in range(len(probs)):
                a = np.random.uniform()
                if a < entropy:
                    print(f"==> ENTROPY: Adding 2% prob to idx {idx}")
                    probs[idx] += ENTROPY_PROB_INC
        
        class_idx_prob_list = []
        for p_idx, pb in enumerate(probs):
            if pb > threshold:
                class_idx_prob_list.append((p_idx, pb))

        # Sort token classes by probability in descending order
        token_classes = sorted(class_idx_prob_list, key=lambda x: x[1], reverse=True)
        
        # Explore token classes with probability > threshold
        for token_class, prob in token_classes:
            # Create new sequence with this token added
            new_sequence = current_sequence + [token_class]

            # Update joint probability
            new_prob = current_prob * prob
            if new_prob < threshold:
                return
                
            recurse(new_sequence, new_prob, depth + 1)
    
    # Start recursion with initial sequence
    recurse(initial_sequence)

    # Walk up the tree and collect all parent instruction indices and calculate joint log-probability of the
    # full program up to this point (which is distinct from the individual instruction step sequence joint
    # probabilities!)
    current_node = selected_node
    joint_log_prob = 0.0
    cumulative_uncertainty = 0.0
    parent_idx_seq = []
    num_steps = 0
    while current_node.parent_node is not None:
        parent_idx_seq.append(current_node.instruction_idx)
        
        # Add the log probability of the instruction that led to this node
        joint_log_prob += current_node.parent_node.log_probs[current_node.instruction_idx]
        cumulative_uncertainty += current_node.parent_node.uncertainties[current_node.instruction_idx]    
        num_steps += 1
        
        current_node = current_node.parent_node

    parent_idx_seq.reverse()

    # now update the selected_node's instruction list
    for seq_idx, instr_seq in enumerate(complete_sequences):
        selected_node.instruction_seqs.append(instr_seq)
        selected_node.uncertainties.append(0.)
        selected_node.log_probs.append(math.log(sequence_probabilities[seq_idx]))

        print(f"==> [{seq_idx}] sequence: {instr_seq}, Prob.: {sequence_probabilities[seq_idx]}")
        
        prog_idx_seq = parent_idx_seq.copy()
        prog_idx_seq.append(seq_idx)

        tmp_log_prob = joint_log_prob + math.log(sequence_probabilities[seq_idx])
        if PRIORITIZATION == 'joint_prob':
            new_prob = tmp_log_prob
        else:
            new_prob = np.exp(tmp_log_prob / (num_steps+1))
        
        # Insert into global_leaf_stats in sorted order based on joint log-probability
        insert_idx = 0
        while insert_idx < len(global_leaf_stats) and global_leaf_stats[insert_idx][0] > new_prob:
            insert_idx += 1

        # if memory_len == MEMORY_LIMIT - 1, we can only add "Delete" actions to global_leaf_stats!
        if memory_len >= MEMORY_LIMIT - 1:
            if instr_seq[1] == 29:
                global_leaf_stats.insert(insert_idx, (new_prob, prog_idx_seq))
        else:
            global_leaf_stats.insert(insert_idx, (new_prob, prog_idx_seq))

def generate_program_state(node, verbose):
    '''
        Runs the program step for the specified node, then gets its output.         
        It also returns that output so that we can check for exit conditions (success or timeout).

        Parameters:
            @param node: the SearchTreeNode being expanded.

        Returns:
            Returns the output of executing the program step.
    '''
    if node.parent_node is None:
        return node.state_variables
    
    instr_idx = node.instruction_idx

    prog_to_run = node.parent_node.instruction_seqs[instr_idx]

    result = [0] # arbitrary non-None state for Delete action
    if prog_to_run[1] != 29: # if not a Delete action, execute it
        # Walk up the tree from current node to root, collecting state variables
        intermediate_state = []
        node_sequence = ts.get_node_sequence(node, DSL_size)
    
        # collect states from node_sequence
        for n in node_sequence:
            intermediate_state.append(n.state_variables)
    
        result = pi.execute_instruction_step(prog_to_run, intermediate_state, DSL)

        if result is None:
            print("==> ERROR in program execution, intermediate state was:")
            for st_idx, st in enumerate(intermediate_state):
                print(f"State #{st_idx}: {st}")
    
    node.state_variables = result

    if result is not None:
        try:
            if state_tokenizer.tokenize(result) is None:
                print(f"==> STATE TOKENIZER: tokenizing {result} failed")
                node.state_variables = None
                return None
        except Exception:
            return None

    return result


def is_goal(state, target):
    if isinstance(state, DSL.Grid):     # TODO: or a list of Grids when there is more than one example pair!
        if np.all(state.cells == target.cells):
            return True

    return False


def select_node(root_node, max_tree_depth):
    global global_leaf_stats

    if len(global_leaf_stats) == 0:
        return None
    
    # We assume this list is already sorted.
    found_valid_path = False
    while not found_valid_path:
        optimal_path = global_leaf_stats[0][1]
        if len(optimal_path) >= max_tree_depth:
            print(f"==> Removing path {optimal_path} because it reached maximum tree depth!")
            global_leaf_stats.pop(0)
            if len(global_leaf_stats) == 0:
                return None
        else:
            found_valid_path = True

    cur_node = root_node
    decoding_step = 0
    for idx in optimal_path:
        if idx in cur_node.child_nodes:
            cur_node = cur_node.child_nodes[idx]
            decoding_step += 1
        else:
            break

    child_node = SearchTreeNode(None, idx, decoding_step+1, cur_node)
    cur_node.child_nodes[idx] = child_node

    # Remove the first element from global_leaf_stats after selecting it
    if global_leaf_stats:
        global_leaf_stats.pop(0)

    return child_node


def search(model, input_grids, target_grids, time_budget, max_iterations, max_tree_depth=MAX_TREE_DEPTH, verbose=False, device='cuda'):
    global global_leaf_stats
    
    '''
    Runs the tree search, looking for the program that can transform the input_grids into their target_grids.

    Fails on timeout.

    Parameters:
        @param model: the pre-trained PyTorch model that predicts token probabilities.
        @param input_grids: the DSL.Grid instance(s) that correspond to the input grid(s) of the problem of solve.
        @param target_grids: the DSL.Grid instance(s) that correspond to the target grid(s) of the problem of solve.
        @param time_budget: the search timeout interval in seconds.
        @param max_iterations: the search's maximum number of iterations allowed.

    Returns:
        (bool, np.ndarray) -- True is solution found, False otherwise. If solution found, the list of 
        token sequences (instruction sequences) representing the solution program.
    '''

    start_time = time.time()
    global_leaf_stats = []
    print("Task target grid: ", target_grids)
    
    root_node = selected_node = SearchTreeNode(input_grids, 0)
    tokenized_targets = np.array(state_tokenizer.tokenize(target_grids))    # TODO: tokenize_batch?
    tokenized_targets = np.reshape(tokenized_targets, [1, tokenized_targets.shape[-1]])
    tokenized_grids_torch = torch.from_numpy(np.array(tokenized_targets, dtype=np.int64)).to(device)
    zero_state_idx = np.zeros((len(tokenized_targets), 1))
    target_grid_mem = model.encode(tokenized_grids_torch)
    target_grid_mem = model.add_state_idx_embed(target_grid_mem, zero_state_idx)
    entropy = 0.0       # start with no entropy
    num_iterations = 0
    while (time.time() - start_time) < time_budget:
        # We start with "selected_node", an empty, un-expanded node to be expanded.

        # Step 1) Run the program step, get its output. Populate the state of the child node, giving
        # it the updated program state based on the parent program step's output. It also returns that output
        # so that we can check for exit conditions (success or timeout)
        output_state = generate_program_state(selected_node, verbose)

        # First confirm that we have a valid program that results in some valid output
        if output_state is not None:

            # Step 2) Check for exit conditions
            if is_goal(output_state, target_grids):
                print("GOAL FOUND!!!")
                return True, selected_node

            if num_iterations > max_iterations:
                return False, None

            # Step 3) generate a distribution of token probabilities
            # from the currently selected node (program state) in the search tree.
            enumerate_programs(model, selected_node, target_grid_mem, entropy=entropy)

        else:

            if verbose:
                print("==> Invalid program, removing it from global list.")

            # If the program execution failed, remove this node from consideration
            # if len(global_leaf_stats) > 0:
            #     global_leaf_stats.pop(0)  # Remove the first element from global_leaf_stats

        # Step 4) Select the next node that maximizes conviction
        # Display the content of global_leaf_stats
        if verbose:
            print("==> Global leaf stats:")
            for i, leaf_stat in enumerate(global_leaf_stats):
                print(f"  Leaf {i}: {leaf_stat}")
        
        selected_node = select_node(root_node, max_tree_depth)
        print("selected_node: ", selected_node)

        if selected_node is None:
            # Here we expanded the entire non-zero probability space and found nothing, but
            # we haven't used up all our resources. Increase entropy.
            root_node = selected_node = SearchTreeNode(input_grids, 0)
            entropy += ENTROPY_INC
            print("==> INREASING ENTROPY!")
            #return False, None
            
        num_iterations += 1

    print("==> TIMEOUT! No solution found.")
    return False, None

