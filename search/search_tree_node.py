
class SearchTreeNode:

    def __init__(self, state_variables, instruction_idx, decoding_step=0, parent_node=None):

        # Indexes into the instruction_seqs, uncertainties and probabilities of the parent node, indicating
        # which instruction sequence was evaluated from the parent node to derive this child node.
        self.instruction_idx = instruction_idx

        # The current program state, represented as Python variables. Here, grids are represented as DSL.Grid
        # instances. This type of data formatting is meant to program execution and checking for a success condition.
        # Note: because this is used for program execution, the initial state variable is not the input + output Grid,
        # but instead only the input Grid.
        self.state_variables = state_variables

        # The current program state, represented as tokenized encoder embedding This formatting is intended to be passed to
        # the PyTorch model's decoder to derive token prediction probabilities. At each decoding step we simply
        # concatenate the encoder output associated with the new state variables from applying the program step.
        self.encoder_memory = None

        self.instruction_seqs = []
        self.log_probs = []
        self.uncertainties = []
        self.parent_node = parent_node
        self.decoding_step = decoding_step
        self.child_nodes = {}

    def __repr__(self):
        # Build a representation of the path from root to this node
        path = []
        current = self
        while current is not None:
            if current.instruction_idx is not None:  # Skip None for root node
                if current.parent_node is None:
                    path.append("root")
                else:
                    path.append(str(current.instruction_idx))
            current = current.parent_node
        
        # Reverse to get path from root to this node
        path.reverse()
        path_str = "->".join(path) if path else "root"
        
        # Return a string representation of this node
        return f"SearchTreeNode(path={path_str})"