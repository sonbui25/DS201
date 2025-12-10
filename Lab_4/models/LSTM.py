import torch
from torch import nn
import torch.nn.functional as F
from utils.vocab import Vocab
class LSTM(nn.Module):
    def __init__(self, vocab: Vocab, config: dict):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.encoder_layers = config['encoder_layers']
        self.decoder_layers = config['decoder_layers']
        
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.get_src_vocab_size(),
            embedding_dim=self.embedding_dim,
            padding_idx=vocab.pad_id,
        )
        
        self.encoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config['dropout'],
        )
        
        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.get_tgt_vocab_size(),
            embedding_dim=2*self.hidden_dim,
            padding_idx=vocab.pad_id,
        )
        self.decoder = nn.LSTM(
            input_size= 2*self.hidden_dim,
            hidden_size=2*self.hidden_dim,
            num_layers=self.decoder_layers,
            batch_first=True,
            bidirectional=False,
            dropout=config['dropout'],
        )
        
        self.output_head = nn.Linear(2*self.hidden_dim, vocab.get_tgt_vocab_size())
        
    def forward_step(self, input_ids: torch.Tensor, 
                     context: torch.Tensor, 
                     dec_hidden_states: torch.Tensor):
        embedded_input = self.tgt_embedding(input_ids)
        dec_hidden_states = dec_hidden_states.contiguous()
        _, (dec_hidden_states, _) = self.decoder(
            embedded_input, (dec_hidden_states, context)
        )
        return dec_hidden_states.contiguous()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        self.train()
        embedded_x = self.src_embedding(x)
        bs, l, hidden_dim = embedded_x.shape
        
        # Pass whole input sequence to Encoder
        # output: (bs, l, 2*hidden_dim), hidden_state: (2*layers, bs, hidden_dim)
        # cell_state: (2*layers, bs, hidden_dim)
        output, (enc_hidden_state, enc_cell_state) = self.encoder(embedded_x) 
        # Reshape hidden state and cell state from Encoder to initialize Decoder
    
        num_directions = 2
        
        #  Split and concat the hidden states from both directions
        # (2*layers, bs, hidden_dim) -> (layers, 2, bs, hidden_dim) -> (layers, bs, 2*hidden_dim)
        enc_hidden_state = enc_hidden_state.view(self.encoder_layers, num_directions, bs, hidden_dim)
        enc_hidden_state = torch.cat((enc_hidden_state[:, 0, :, :], enc_hidden_state[:, 1, :, :]), dim=2)
        
        # (2*layers, bs, hidden_dim) -> (layers, 2, bs, hidden_dim) -> (layers, bs, 2*hidden_dim)
        enc_cell_state = enc_cell_state.view(self.encoder_layers, num_directions, bs, hidden_dim)
        enc_cell_state = torch.cat((enc_cell_state[:, 0, :, :], enc_cell_state[:, 1, :, :]), dim=2)

        # Initialize Decoder hidden state and cell state from Encoder hidden state and cell state with proper layers
        cell_mem = enc_cell_state[:self.decoder_layers].contiguous() # cell state or Long-term memory from encoder , shape: (layers, bs, 2*dim)
        dec_H_init = enc_hidden_state[:self.decoder_layers].contiguous() # hidden state or Short-term memory from encoder , shape: (layers, bs, 2*dim)
        
        _, tgt_len = y.shape
        logits = []
        
        # 3. Decoder loop
        current_dec_hidden_state = dec_H_init
        
        for ith in range(tgt_len):
            y_ith = y[:, ith].unsqueeze(-1)
            current_dec_hidden_state = self.forward_step(y_ith, cell_mem, current_dec_hidden_state)
            
            # get the last hidden states
            last_hidden_state = current_dec_hidden_state[-1] # Get the final layer of the final time step t hidden state (bs, dim)
            logit = self.output_head(last_hidden_state)
            logits.append(logit.unsqueeze(1))
        
        logits = torch.cat(logits, dim=1)
        
        return logits
    
    def predict(self, x: torch.Tensor):
        self.eval()
        embedded_x = self.src_embedding(x)
        bs, l, _ = embedded_x.shape
        
        outputs, (enc_hidden_state, enc_cell_state) = self.encoder(embedded_x)
        
        num_directions = 2
        
        enc_hidden_state = enc_hidden_state.view(self.encoder_layers, num_directions, bs, self.hidden_dim)
        enc_hidden_state = torch.cat((enc_hidden_state[:, 0, :, :], enc_hidden_state[:,1, :, :]), dim=2)
        
        enc_cell_state = enc_cell_state.view(self.encoder_layers, num_directions, bs, self.hidden_dim)
        enc_cell_state = torch.cat((enc_cell_state[:, 0, :, :], enc_cell_state[:,1, :, :]), dim=2)
        
        dec_H_init = enc_hidden_state[:self.decoder_layers].contiguous()
        current_dec_hidden_state = dec_H_init

        cell_mem = enc_cell_state[:self.decoder_layers].contiguous()
        mark_eos = torch.zeros_like(y_ith).bool()
        outputs = []
        while True:
            current_dec_hidden_state = self.forward_step(y_ith, cell_mem, current_dec_hidden_state)
            # get the last hidden states
            last_hidden_state = current_dec_hidden_state[-1]
            logit = self.output_head(last_hidden_state)
            
            y_ith = logit.argmax(dim=-1).long() 
            mark_eos = (y_ith == self.vocab.eos_id)
            
            if all(mark_eos.tolist()):
                break
            outputs.append(y_ith.unsqueeze(-1)) 
            
        outputs = torch.cat(outputs, dim=-1) 
        return outputs
        