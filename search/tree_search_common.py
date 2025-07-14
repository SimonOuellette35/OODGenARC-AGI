import torch
from state_tokenizer import StateTokenizer
import numpy as np
from AmotizedDSL.prog_utils import ProgUtils


# this function starts from a given child node, and obtains all the parent nodes by walking
# up the tree. It then sorts them in top-down order (i.e. normal program sequence) and applies
# any delete actions. This returns the sequence of states (as nodes) that are relevant at the
# end of this partial program's execution. Useful for prepare_encoder_memory() and for
# generate_program_state().
def get_node_sequence(node, DSL_size, DSL):
    node_sequence = []
    cur_node = node
    while cur_node.parent_node is not None:
        cur_node = cur_node.parent_node
        node_sequence.append(cur_node)

    node_sequence.reverse()
    node_sequence.append(node)

    # NOTE: the confusing this about this is that node 0's state is the input grid, and its
    # instruction seq is the first instruction to be executed given the input grid. Node 1's
    # state is the output of applying instruction 0, while it's instruction is the second
    # instruction to be executed, etc.
    
    del_token_id = DSL.prim_indices['del'] + ProgUtils.NUM_SPECIAL_TOKENS

    # Process each node in sequence, checking for delete operations
    i = 0
    while i < len(node_sequence):
        current_node = node_sequence[i]

        if current_node.parent_node is not None:
            instr_seq = current_node.parent_node.instruction_seqs[current_node.instruction_idx]
            
            # Check if it's a delete operation
            if instr_seq[0] == 0 and instr_seq[1] == del_token_id:
                # Get the state index to delete from the next token in the sequence
                state_idx_to_del = instr_seq[3]
                state_idx_to_del -= DSL_size
                print(f"==> DELETING @ state_idx_to_del = {state_idx_to_del} -- current idx {i}, current node_sequence len = {len(node_sequence)}")
                
                # Remove the item at that index from node_sequence
                if state_idx_to_del < i and state_idx_to_del >= 0:
                    # Also delete the current, actual delete node
                    del node_sequence[i]
                    del node_sequence[state_idx_to_del]
    
                    i -= 1
                    continue
        
        # Move to the next node
        i += 1

    return node_sequence


def prepare_encoder_memory(model, node, target_memory, state_tokenizer, DSL_size, DSL, device='cuda'):
    model.eval()
    with torch.no_grad():

        tokenized_state = np.array(state_tokenizer.tokenize(node.state_variables))
        
        # Pad all sequences in tokenized_batch_output to max_seq_length
        tokenized_state = StateTokenizer.pad(tokenized_state)

        print("==> Using tokenized (padded) state sequence: ", list(tokenized_state))

        tokenized_state_torch = torch.unsqueeze(torch.from_numpy(tokenized_state).to(device), dim=0)

        latent_state = model.encode(tokenized_state_torch)

        node.encoder_memory = latent_state.cpu().data.numpy()

        node_sequence = get_node_sequence(node, DSL_size, DSL)
        
        print("==> Preparing encoder memory from state sequence:")
        enc_memories_torch = []
        for n_idx, n in enumerate(node_sequence):
            print(f"\t#{n_idx}: {n.state_variables}")
            enc_memory = torch.from_numpy(n.encoder_memory).to(device)
            
            # target grid is state_idx 0, actual state indices are 1-indexed.
            state_idx = np.ones((1, 1), dtype=np.int64) * (n_idx + 1)
            
            enc_memory = model.add_state_idx_embed(enc_memory, state_idx)
            enc_memories_torch.append(enc_memory)

        # Prepend target_memory to enc_memories_torch
        # target_memory is already encoded with state_idx 0
        enc_memories_torch.insert(0, target_memory)

        # Concatenate all tensors along dimension 1
        encoder_memory = torch.cat(enc_memories_torch, dim=1)

        # Pad encoder memory with zeros along the second dimension
        # batch_size, seq_len, hidden_dim = encoder_memory.shape
        # pad_length = 204
        # if seq_len < pad_length:
        #     padding_size = pad_length - seq_len
        #     padding = torch.zeros(batch_size, padding_size, hidden_dim, device=encoder_memory.device)
        #     encoder_memory = torch.cat([encoder_memory, padding], dim=1)

    return encoder_memory, len(node_sequence)
