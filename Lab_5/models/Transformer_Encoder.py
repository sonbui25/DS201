import math
import torch
from torch import nn
from utils.vocab import Vocab

def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.size(1)
        mask = torch.arange((maxlen), dtype=torch.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a large negative value
        X = _sequence_mask(X.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class DotProductAttention(nn.Module):
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None  # Initialize attention_weights
    
    # Shape of queries: (batch_size, num_queries, num_hiddens)
    # Shape of keys: (batch_size, num_kv, num_hiddens)
    # Shape of values: (batch_size, num key-value pairs, value_dim)
    # Shape of valid_lens: (batch_size) or (batch_size, num_queries)               
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        #Swap the last two dimensions of keys with keys.transpose(1,2)
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        return torch.bmm(self.dropout(self.attention_weights), values)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_k = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_v = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self.W_0 = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        self._init_xavier()
    
    def _init_xavier(self):
        """Initialize weights with Xavier initialization."""
        for module in [self.W_q, self.W_k, self.W_v, self.W_0]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, queries, keys, values, valid_lens):
        # Shape of queries, keys, values:
        # (batch_size, num_querry or num_kv, num_hiddens)
        # Shape of valid_lens:
        # (batch_size) or (batch_size, num_queries)
        # After transposing, the shape of output queries, keys, values:
        # (batch_size * num_heads, num_queries or num_kv, num_hiddens / num_heads)
        queries = self.tranpose_qkv(self.W_q(queries))
        keys = self.tranpose_qkv(self.W_k(keys))
        values = self.tranpose_qkv(self.W_v(values))
        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads times, then copy the next item, and so on
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
        
        output = self.attention(queries, keys, values, valid_lens) #(bs*num_heads, num_queries or num_kv, num_hiddens / num_heads)
        self.attention_weights = self.attention.attention_weights  # Store attention weights
        output_concat = self.transpose_output(output) #(bs, num_queries or num_kv, num_hiddens)
        return self.W_0(output_concat)
        
    def tranpose_qkv(self, X):
        """Transposition for parallel computation of multiple attention heads.""" 
        # Shape of input X: (bs, num of querry or kv, num_hiddens) -> output X: (bs * num_heads, num of querry or kv, num_hiddens / num_heads)
        X = X.reshape(X.shape[0], X.shape[1], self.num_heads, -1)# (bs, num_heads, num of querry or kv, num_hiddens / num_heads) -> X.shape[3] will be calculated automatically as -1 for (num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3) # (bs, num_heads, num of querry or kv, num_hiddens / num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])#(bs * num_heads, num of querry or kv, num_hiddens / num_heads)
    def transpose_output(self, X):
        """Reverse the transposition operation of multi-head attention."""
        X = X.reshape(-1, self.num_heads, X.shape[1], X.shape[2]) # (bs, num_heads, num of querry or kv, num_hiddens / num_heads)
        X = X.permute(0, 2, 1, 3) # (bs, num of querry or kv, num_heads, num_hiddens / num_heads)
        return X.reshape(X.shape[0], X.shape[1], -1) # (bs, num of querry or kv, num_hiddens)

class PositionEncoding(nn.Module):
    """Positional encoding module."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.register_buffer('P', torch.zeros((1, max_len, num_hiddens)))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
                0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        
        self.P[:, :, 0::2] = torch.sin(X) # :: means start from 0 to the end, step by 2
        self.P[:, :, 1::2] = torch.cos(X) # :: means start from 1 to the end, step by 2
        
    def forward(self, X):
        X = X + self.P[: , :X.shape[1], :].to(X.device) # Plus P to the first actual length of X
        return self.dropout(X)

class PositionWiseFFN(nn.Module):
    # Thêm tham số input_size để xử lý đúng chiều dữ liệu
    def __init__(self, ffn_num_hiddens, ffn_num_outputs, input_size):
        super().__init__()
        # Input layer: Từ input_size (d_model) -> ffn_num_hiddens (thường lớn hơn, vd 4*d_model)
        self.dense1 = nn.Linear(input_size, ffn_num_hiddens)
        self.relu = nn.ReLU()
        # Output layer: Từ ffn_num_hiddens -> ffn_num_outputs (d_model)
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)
            
    def forward(self, x):
        return self.dense2(self.relu(self.dense1(x)))
    
class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)
        
    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens, input_size=num_hiddens)
        
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        
    def forward(self, X, valid_lens):
        X = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        X = self.addnorm2(X, self.ffn(X))
        return X
class TransformerEncoder(nn.Module):

    def __init__(self, vocab: Vocab, config: dict):

        super(TransformerEncoder, self).__init__()
        self.embedding_dim = 256
        self.vocab_size = vocab.get_vocab_size
        self.hidden_dim = 256

        self.num_layers = 3

        self.num_classes = vocab.total_labels
        self.task_type = config['vocab']['task_type']

        self.dropout = 0.1

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=vocab.pad_id)

        self.pos_encoding = PositionEncoding(self.embedding_dim, self.dropout)
        self.blks = nn.Sequential()
        for i in range(self.num_layers):
            self.blks.add_module(f"block_{i}", TransformerEncoderBlock(
                num_hiddens=self.hidden_dim,
                ffn_num_hiddens=256,
                num_heads=4,
                dropout=self.dropout,
                use_bias=False
            ))

        self.fc = nn.Linear(self.hidden_dim, self.num_classes)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, X, valid_lens):

        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.hidden_dim))
        self.attention_weights = [None] * self.num_layers
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention_weights
        X = self.dropout_layer(X)
        X = self.fc(X)  # Shape: (bs, seq_len, num_classes)
        
        # Return appropriate shape based on task type
        if self.task_type == 'text_classification':
            # For text_classification: take first token [CLS] -> (bs, num_classes)
            X = X[:, 0, :]
        # For seq_labeling: return full sequence (bs, seq_len, num_classes)
        
        return X