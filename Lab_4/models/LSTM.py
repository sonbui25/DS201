import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
import random

class LSTM(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #  Dimensions 
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.encoder_layers = config['encoder_layers']
        self.decoder_layers = config['decoder_layers']
        self.dropout_rate = config['dropout']
        
        #  Embedding 
        self.src_embedding = nn.Embedding(
            num_embeddings=vocab.get_src_vocab_size(),
            embedding_dim=self.embedding_dim,
            padding_idx=vocab.pad_id
        )
        
        self.tgt_embedding = nn.Embedding(
            num_embeddings=vocab.get_tgt_vocab_size(),
            embedding_dim=self.embedding_dim,
            padding_idx=vocab.pad_id
        )

        #  1. ENCODER (Bidirectional) 
        self.encoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.encoder_layers,
            batch_first=True,
            bidirectional=True, 
            dropout=self.dropout_rate if self.encoder_layers > 1 else 0
        )
        
        self.enc_output_dim = self.hidden_dim * 2 # Bidirectional
        
        #  2. DECODER (Multi-layer LSTMCells) 
        self.decoder_cells = nn.ModuleList()
        
        for i in range(self.decoder_layers):
            # Layer 0: Nhận Embedding + Context Vector
            if i == 0:
                input_size = self.embedding_dim + self.enc_output_dim
            else:
                input_size = self.enc_output_dim
            
            self.decoder_cells.append(
                nn.LSTMCell(
                    input_size=input_size, 
                    hidden_size=self.enc_output_dim 
                )
            )

        self.fc_out = nn.Linear(self.enc_output_dim, vocab.get_tgt_vocab_size())
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x, y=None):
        """
        x: [batch, src_len]
        y: [batch, tgt_len] (Optional)
        Nếu y có thì dùng để Teacher Forcing, không thì dùng để Inference
        """
        batch_size = x.shape[0]
        
        #  BƯỚC 1: TÍNH ĐỘ DÀI CÂU (Bỏ qua Padding) 
        # Tính số lượng token khác PAD trong mỗi câu
        src_lens = torch.sum(x != self.vocab.pad_id, dim=1).cpu()
        
        #  BƯỚC 2: ĐẢO NGƯỢC CÂU NGUỒN (REVERSE SOURCE) 
        # Chỉ đảo phần từ thật, giữ nguyên padding ở cuối
        x_reversed = x.clone()
        for i, length in enumerate(src_lens):
            # Chỉ đảo ngược đoạn [0 : length] của từng câu
            x_reversed[i, :length] = torch.flip(x[i, :length], dims=[0])
            
        #  BƯỚC 3: ENCODER (Với Packed Sequence) 
        embedded_x = self.dropout(self.src_embedding(x_reversed))
        # Nén input: Giúp LSTM bỏ qua tính toán trên các token PAD
        packed_embedded = pack_padded_sequence(
            embedded_x, 
            src_lens, 
            batch_first=True, 
            enforce_sorted=False 
        )
        # Chạy Encoder
        packed_outputs, (hidden, cell) = self.encoder(packed_embedded)
        
        #  BƯỚC 4: LẤY CONTEXT VECTOR 
        # Lấy hidden state cuối cùng (đại diện cho toàn bộ câu nguồn)
        # hidden shape: [2 * layers, batch, hidden_dim]
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        context = torch.cat((hidden_fwd, hidden_bwd), dim=1)  # [batch, hidden_dim * 2]
        
        #  PREPARE DECODER STATES 
        # Reshape lại để khớp với Decoder (Layer by Layer)
        hidden = hidden.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        
        # Gộp chiều Fwd và Bwd
        dec_hidden_states = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        dec_cell_states = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        
        h_states = [dec_hidden_states[i] for i in range(self.decoder_layers)]
        c_states = [dec_cell_states[i] for i in range(self.decoder_layers)]

        #  DECODER LOOP 
        outputs = []
        
        if y is not None:
            max_len = y.shape[1]
            decoder_input = y[:, 0] 
        else:
            max_len = self.config['max_length']
            decoder_input = torch.tensor([self.vocab.bos_id] * batch_size, device=x.device)

        num_steps = range(1, max_len) if y is not None else range(max_len)
        
        for t in num_steps:
            # 1. Embed Input
            emb_input = self.dropout(self.tgt_embedding(decoder_input))
            
            # 2. Context Injection: Nối Context vào Input Decoder layer 0
            emb_and_context = torch.cat((emb_input, context), dim=1)
            
            # 3. Chạy các layer LSTM
            h_states[0], c_states[0] = self.decoder_cells[0](emb_and_context, (h_states[0], c_states[0]))
            curr_input = h_states[0]
            
            for i in range(1, self.decoder_layers):
                h_states[i], c_states[i] = self.decoder_cells[i](curr_input, (h_states[i], c_states[i]))
                curr_input = h_states[i]
            
            # 4. Prediction
            prediction = self.fc_out(h_states[-1])
            outputs.append(prediction.unsqueeze(1))
            
            # 5. Chọn input tiếp theo
            if y is not None:
                decoder_input = y[:, t] 
            else:
                decoder_input = prediction.argmax(1)
                if (decoder_input == self.vocab.eos_id).all():
                    break

        logits = torch.cat(outputs, dim=1)
        return logits