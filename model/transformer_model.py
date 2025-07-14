import torch
import torch.nn as nn
import numpy as np
import math
import json


# Positional Encoding function
def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)

# RoPE positional encoding (Rotary Position Encoding)
# TODO: I'm not sure this is implemented correctly because it doesn't seem to perform as 
# well as the standard positional encoding above. Slower training convergence.
def get_RoPE_embeddings(seq_len, dim):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
    sinusoid_inp = torch.einsum("ij,k->ik", pos, inv_freq)
    embeddings = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
    return embeddings

def apply_RoPE(x, positional_embeddings):
    # Slice positional embeddings to match input sequence length
    seq_len = x.size(1)
    positional_embeddings = positional_embeddings[:seq_len, :]
    # Expand positional embeddings to match batch dimension
    positional_embeddings = positional_embeddings.expand(x.size(0), -1, -1)
    # Element-wise multiplication
    x_rotated = x * positional_embeddings
    return x_rotated

class StandardTransformerModel(nn.Module):

    @staticmethod
    def instantiate_from_config_file(config_file, device='cuda'):
        """
        Instantiate a StandardTransformerModel from a JSON configuration file.
        
        Args:
            config_file (str): Path to the JSON configuration file
            
        Returns:
            StandardTransformerModel: An instance of the model with parameters from the config file
        """
        
        # Load configuration from JSON file
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract parameters from config
        d_model = config['d_model']
        num_heads = config['num_heads']
        num_layers = config['num_layers']
        input_vocab_size = config['input_vocab_size']
        target_vocab_size = config['target_vocab_size']
        max_state_seq_length = config['max_state_seq_length']
        max_target_seq_length = config['max_instr_seq_length']
        dropout = config['dropout']
        max_instr_steps = config['max_instr_steps']
        dim_feedforward = config['dim_feedforward']

        # Create and return model instance
        return StandardTransformerModel(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers=num_layers,
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            max_state_seq_length=max_state_seq_length,
            max_target_seq_length=max_target_seq_length,
            max_instr_steps=max_instr_steps,
            dropout=dropout,
            device=device
        )        

    def __init__(self, d_model, dim_feedforward, num_heads, num_layers, input_vocab_size, target_vocab_size, max_state_seq_length,
                 max_target_seq_length, max_instr_steps, 
                 use_rope_embedding=False, device='cuda', dropout=0.1, spectral_decoupling=False):
        super(StandardTransformerModel, self).__init__()
        print(f"==> input vocab size: {input_vocab_size}")
        print(f"==> target vocab size: {target_vocab_size}")

        self.max_instr_steps = max_instr_steps
        self.max_state_seq_length = max_state_seq_length
        self.max_target_seq_length = max_target_seq_length
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.device = device
        self.spectral_decoupling = spectral_decoupling
        self.use_rope_embedding = use_rope_embedding
        self.d_model = d_model

        # Initialize embeddings with better scaling
        self.index_embedding = nn.Embedding(20, d_model)
        self.src_embedding = nn.Embedding(input_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(target_vocab_size, d_model)
        
        # Initialize embeddings with Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.src_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.tgt_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.index_embedding.weight, gain=1.0)

        # Scale embeddings by sqrt(d_model)
        self.embed_scale = math.sqrt(d_model)
        
        if self.use_rope_embedding:
            self.state_pos_encoding = get_RoPE_embeddings(max_state_seq_length+1, d_model).to(device)
            self.target_pos_encoding = get_RoPE_embeddings(max_target_seq_length, d_model).to(device)
        else:
            self.state_pos_encoding = positional_encoding(max_state_seq_length+1, d_model).to(device)
            self.target_pos_encoding = positional_encoding(max_target_seq_length, d_model).to(device)

        # Create masks once during initialization
        self.causal_mask = torch.triu(torch.ones(max_target_seq_length, max_target_seq_length), diagonal=1).bool()
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Create encoder and decoder with improved parameters
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',  # Using GELU activation instead of ReLU
            batch_first=True,
            norm_first=True  # Pre-normalization for better training stability
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        # Add layer normalization
        encoder_norm = nn.LayerNorm(d_model)
        decoder_norm = nn.LayerNorm(d_model)
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers, norm=decoder_norm)
        
        self.fc = nn.Linear(d_model, target_vocab_size)
        # Initialize output projection with Xavier/Glorot
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Add label smoothing

    def add_state_idx_embed(self, memory, batch_state_idx):
        # Add the embedding representing the intermediate state index
        np_state_idx = (batch_state_idx).astype(np.int64)
        state_idx_tensor = torch.from_numpy(np_state_idx).to(self.device)
        state_idx_embed = self.index_embedding(state_idx_tensor)

        memory += state_idx_embed

        return memory

    def encode(self, src):
        src_emb = self.src_embedding(src) * self.embed_scale
        
        if self.use_rope_embedding:
            src_emb = apply_RoPE(src_emb, self.state_pos_encoding)
        else:
            src_emb = src_emb + self.state_pos_encoding[:, :src_emb.size(1), :]
        
        # Create padding mask for encoder
        src_padding_mask = (src == 0)

        src_emb = self.dropout(src_emb)

        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)

        return memory
    
    def decode(self, memory, tgt, use_teacher_forcing=True):
        if use_teacher_forcing:
            tgt_input = tgt[:, :-1]
        else:
            tgt_input = tgt

        # Create causal mask for decoder
        tgt_mask = self.causal_mask[:tgt_input.size(1), :tgt_input.size(1)].to(self.device)

        # Display range of token values in tgt_input
        # if tgt_input.numel() > 0:
        #     min_token = tgt_input.min().item()
        #     max_token = tgt_input.max().item()
        #     print(f"Token range in tgt_input: min={min_token}, max={max_token}")
        # else:
        #     print("tgt_input is empty")

        # Embedding and positional encoding with scaling
        # Display range of token values in tgt_input
        tgt_emb = self.tgt_embedding(tgt_input) * self.embed_scale
        
        if self.use_rope_embedding:
            tgt_emb = apply_RoPE(tgt_emb, self.target_pos_encoding)
        else:
            tgt_emb = tgt_emb + self.target_pos_encoding[:, :tgt_emb.size(1), :]
        
        # Apply dropout after embedding and positional encoding
        tgt_emb = self.dropout(tgt_emb)

        # Transformer layers
        # Create padding mask for memory (encoder output)
        memory_padding_mask = (memory == 0).all(dim=-1)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_padding_mask)
        
        return self.fc(output)

    # Auto-regressively generate the predictions without teacher forcing.
    def generate(self, encoder_memory, SOS_token, init_sequence=None, iter_max=None):
        batch_size = encoder_memory.size(0)
    
        # Start with SOS token
        if init_sequence is not None:
            # Convert the list of integers to a tensor and add to the decoded sequence
            init_tensor = torch.tensor(init_sequence, dtype=torch.long, device=self.device).unsqueeze(0).expand(batch_size, -1)
            decoded = torch.cat([torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=self.device), init_tensor], dim=1)
        else:
            decoded = torch.full((batch_size, 1), SOS_token, dtype=torch.long, device=self.device)

        if iter_max is None:        
            max_gen_length = self.max_target_seq_length
        else:
            max_gen_length = iter_max + 1
        
        # Generate tokens auto-regressively
        probs = []
        for _ in range(max_gen_length - 1):  # -1 because we already have SOS
            # Forward pass through the model
            output = self.decode(encoder_memory, decoded, use_teacher_forcing=False)
            
            # Get the most likely next token
            next_token_logits = output[:, -1, :]
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=1)
            probs.append(next_token_probs)
            next_token = next_token_probs.argmax(dim=1)
            
            # Append the predicted token to the sequence
            decoded = torch.cat([decoded, next_token.unsqueeze(1)], dim=1)
        
        return decoded, probs

    # Pred_seqs is a list of prediction token tensors, of length equal to the number
    # of decoding steps. Each element is a tensor of instruction step token sequences
    # for each task in the batch. So we iterate over decoding steps, not batch elements.
    def loss(self, pred_seqs, target_instr):
        '''
        Loss function.

        Parameters:
            @param pred_seqs: predicted logits, a list of length equal to the number of decoding steps.
                              Each element of the list is a tensor of shape [batch_size, instr sequence length,
                              target vocab size]
            @param target_instr: ground truth programs, shape [batch_size, num decoding steps, inst sequence length]
        '''
        total_loss = 0.

        for instr_step_idx in range(len(pred_seqs)):
            # current_pred_seq = [batch_size, instr sequence length, target vocab size]
            current_pred_seq = pred_seqs[instr_step_idx]
            
            # tgt = [batch_size, instr sequence length]
            tgt = target_instr[:, instr_step_idx, 1:]

            # Reshape output and target for loss computation
            preds_flat = current_pred_seq.contiguous().view(-1, current_pred_seq.shape[-1])
            target_flat = tgt.contiguous().view(-1)
        
            total_loss += self.criterion(preds_flat, target_flat)

        if self.spectral_decoupling:
            SD_loss = 0.

            # Add spectral decoupling regularization term: λ||f(x)||²
            # This helps prevent gradient starvation by encouraging learning all features
            lambda_coeff = 0.001  # Regularization strength, can be tuned
            for pred_logit in pred_seqs:
                normed_logits = torch.norm(torch.reshape(pred_logit, [-1, pred_logit.shape[-1]]), dim=-1)
                l2_term = torch.mean(normed_logits**2)               
                SD_loss += l2_term

            total_loss += lambda_coeff * SD_loss

        return total_loss / float(len(pred_seqs))
    
    def get_encoder_memory(self, tokenized_states, tokenized_targets):
        encoder_memory = None
        batch_state_idx = np.zeros(len(tokenized_states))
        
        tokenized_grids_torch = torch.from_numpy(np.array(tokenized_targets, dtype=np.int64)).to(self.device)
        zero_state_idx = np.zeros((len(tokenized_targets), 1))
        target_grid_mem = self.encode(tokenized_grids_torch)
        target_grid_mem = self.add_state_idx_embed(target_grid_mem, zero_state_idx)
        
        # Calculate the maximum number of states
        max_state_len = 0
        for states in tokenized_states:
            max_state_len = max(max_state_len, len(states))
        
        for step_idx in range(max_state_len):
            # Find which batch entries have states for this step
            valid_batch_indices = []
            valid_states = []
            valid_state_indices = []
            for batch_idx in range(len(tokenized_states)):
                if step_idx < len(tokenized_states[batch_idx]):
                    current_state = tokenized_states[batch_idx][step_idx]

                    if current_state[0] != -1 and len(current_state) > 0:
                        batch_state_idx[batch_idx] = step_idx + 1   # target is state idx 0, the rest is +1

                        valid_batch_indices.append(batch_idx)
                        valid_state_indices.append(batch_state_idx[batch_idx])
                        valid_states.append(current_state)
            
            # If we have any valid states for this step
            if valid_states:
                tokenized_batch_output_torch = torch.from_numpy(np.array(valid_states, dtype=np.int64)).to(self.device)               
                latent_state = self.encode(tokenized_batch_output_torch)

                batched_state_idx = np.reshape(np.array(valid_state_indices), [len(valid_state_indices), 1])
                latent_state = self.add_state_idx_embed(latent_state, batched_state_idx)
                
                d_model = latent_state.shape[-1]

                # Initialize encoder_memory if it's the first step
                if encoder_memory is None:
                    # Create empty tensor with correct batch size    
                    encoder_memory = torch.zeros(len(tokenized_states), 0, d_model, device=self.device)
                
                # For each batch, either add the new latent state or pad with zeros
                new_memories = []
                latent_idx = 0
                
                for batch_idx in range(len(tokenized_states)):
                    if batch_idx in valid_batch_indices:
                        # Add the actual encoded state for this batch item
                        batch_latent = latent_state[latent_idx:latent_idx+1]
                        latent_idx += 1
                    else:
                        # Create a zero tensor with the same shape as a latent state
                        batch_latent = torch.zeros(1, latent_state.shape[1], d_model, device=self.device)
                    
                    if encoder_memory[batch_idx].shape[0] == 0:
                        new_memories.append(torch.cat((target_grid_mem[batch_idx:batch_idx+1], batch_latent), dim=1))
                    else:
                        new_memories.append(torch.cat((encoder_memory[batch_idx:batch_idx+1], batch_latent), dim=1))
                
                encoder_memory = torch.cat(new_memories, dim=0)

        return encoder_memory
