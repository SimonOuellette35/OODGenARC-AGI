import torch
import AmotizedDSL.DSL as DSL
from typing import List, Tuple
import ARC_gym.utils.tokenization as tok
import numpy as np
from AmotizedDSL.program_interpreter import DeleteAction


# We want the StateTokenizer vocabulary to align with ARC_gym tokenization when it comes to grids.
class StateTokenizer:
    MAX_SEQ_LENGTH = 991 # should be 30x31 because during program execution for shifts, the grid can get bigger
                         # with the end of row tokens, that means (30 x (31 + 1)) + 1 = 961

    """Custom tokenizer for sequences of integers (0-9) and special tokens."""
    def __init__(self):
        self.vocab = {
            "PAD": 0,
            "NEWLINE": 1,
            "EOS": 2,
        }
        for i in range(10): # colors 0-9
            self.vocab[str(i)] = i + 3

        self.vocab["SOS"] = 13

        for i in range(10, 31): # integers 10-30 for the grid dimensions
            self.vocab[str(i)] = i + 4

        self.vocab["TRUE"] = 35
        self.vocab["FALSE"] = 36

        # negative values from -1 to -10
        for i in range(1, 11):
            self.vocab[str(-i)] = i + 36
            
        # Type hints
        self.type_hints = {
            'GRID': 47,
            'GRID_LIST': 48,
            'INT': 49,
            'INT_LIST': 50,
            'BOOL': 51,
            'BOOL_LIST': 52
        }
        self.vocab.update(self.type_hints) # Add type hints

        self.vocab_size = len(self.vocab.keys())

    @staticmethod
    def pad(state, length=MAX_SEQ_LENGTH):
        """
        Pad a tokenized state with zeros until it reaches the specified length
        
        Args:
            state: Tokenized state as a numpy array
            length: Target length for padding (default: 101)
        
        Returns:
            Padded state as a numpy array
        """
        # If state is a numpy array, convert to list first
        if isinstance(state, np.ndarray):
            state = state.tolist()

        current_length = len(state)
        if current_length < length:
            padding_needed = length - current_length
            padded_state = np.pad(state, (0, padding_needed), 'constant', constant_values=0)
            return padded_state
        
        return np.array(state)

    def encode(self, sequence, sequence_type, max_length=MAX_SEQ_LENGTH):
        """Encodes a sequence of integers + special tokens into token IDs. Each sequence ends with EOS before padding."""
        type_token_id = self.vocab.get(sequence_type.upper(), self.vocab["INT"])
        tokens = [type_token_id]
        tokens.extend([self.vocab.get(str(token), self.vocab["PAD"]) for token in sequence])
        tokens = tokens[:max_length - 1]  # Truncate to max_length - 1 to make space for EOS
        tokens.append(self.eos_token_id) # Add EOS token
        tokens = tokens + [self.pad_token_id] * max(0, max_length - len(tokens)) # Pad to max_length
        return torch.tensor(tokens)

    def batch_encode(self, sequences, max_length=940, sequence_types=None):
        """Encodes a batch of sequences."""
        if sequence_types is None:
            sequence_types = ["INT"] * len(sequences)
        return torch.stack([self.encode(seq, max_length, seq_type) for seq, seq_type in zip(sequences, sequence_types)])

    def decode(self, token_ids):
        """Decodes token IDs back into sequences of integers."""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        return [reverse_vocab.get(token_id.item(), "UNK") if hasattr(token_id, 'item') else reverse_vocab.get(token_id, "UNK") for token_id in token_ids]

    def tokenize_batch(self, variable_batch):
        tokenized_sequences = []

        for i, variable in enumerate(variable_batch):
            try:
                if variable is None:    # happens when we reach the padding instruction steps (all EOS tokens)
                    tokenized_sequences.append([self.vocab["PAD"]])
                elif isinstance(variable, DeleteAction):
                    tokenized_sequences.append([-1])
                else:
                    tmp = self.tokenize(variable)
                    if tmp is None:
                        print(f"Warning: Could not tokenize variable of type {type(variable)}. Contents: {variable}")
    
                    tokenized_sequences.append(tmp)
            except Exception as e:
                print(f"Error tokenizing batch element {i}: {e}")
                # Display the full traceback for debugging
                import traceback
                traceback_str = traceback.format_exc()
                print(f"Full traceback:\n{traceback_str}")
                # Return None to indicate tokenization failure - the caller needs to handle this
                exit(0) # TODO: temporary
                return None

        return tokenized_sequences

    def tokenize(self, variable):
        if isinstance(variable, DSL.Grid):
            # Reserve one space for the type hint token
            sequence = tok.tokenize_grid(variable.cells, max_length=StateTokenizer.MAX_SEQ_LENGTH-1)
            sequence = [self.type_hints["GRID"]] + list(sequence)
            return sequence
        
        elif isinstance(variable, List) and isinstance(variable[0], DSL.Grid):
            tokenized_grids = []
            for i, grid in enumerate(variable):
                sequence = tok.tokenize_grid(grid.cells, max_length=StateTokenizer.MAX_SEQ_LENGTH)
                
                # Replace 0-padding with EOS tokens for all but the last grid
                if i < len(variable) - 1:
                    # Find where padding starts (first occurrence of 0)
                    padding_start = next((j for j, token in enumerate(sequence) if token == 0), len(sequence))
                    # Replace all 0s with EOS token
                    for j in range(padding_start, len(sequence)):
                        sequence[j] = self.vocab["EOS"]
                
                tokenized_grids.append(sequence)
            
            # Prepend GRID_LIST type hint
            tokenized_grids = [self.type_hints["GRID_LIST"]] + [token for grid_tokens in tokenized_grids for token in grid_tokens] # flatten the list of lists
            return tokenized_grids

        elif isinstance(variable, List) and (isinstance(variable[0], bool) or isinstance(variable[0], np.bool_)):
            tokenized_bools = []
            for b in variable:
                if b:
                    tokenized_bools.append(self.vocab["TRUE"])
                else:
                    tokenized_bools.append(self.vocab["FALSE"])

            # Prepend BOOL_LIST type hint
            tokenized_bools = [self.type_hints["BOOL_LIST"]] + tokenized_bools

            return tokenized_bools

        elif isinstance(variable, List) and isinstance(variable[0], (int, np.int32, np.int64)):
            sequence = [self.type_hints["INT"]]
            for x in variable:
                if str(x) not in self.vocab:
                    print("==> ERROR: StateTokenizer vocab doesn't know about value: ", str(x))
                    exit(-1)
                
                sequence.append(self.vocab[str(x)])

            return sequence

        elif isinstance(variable, (int, np.integer)):
            if str(variable) not in self.vocab:
                print("==> ERROR: StateTokenizer vocab doesn't know about value: ", str(variable))
                exit(-1)

            sequence = [self.type_hints["INT"]] + [self.vocab[str(variable)]]

            return sequence
            
        elif isinstance(variable, (bool, np.bool_)):
            if variable:
                sequence = [self.type_hints["BOOL"]] + [self.vocab["TRUE"]]
            else:
                sequence = [self.type_hints["BOOL"]] + [self.vocab["FALSE"]]
            return sequence

        else:
            print("==> ERROR: unknown variable type for variable: ", variable)
            return None

