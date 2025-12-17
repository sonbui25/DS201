import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# Bahdanau Attention based on "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
class LSTM_Bahdanau(nn.Module):
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
        
        # Encoder Output Dim = hidden_dim * 2 (do 2 chiều: Forward + Backward)
        self.enc_output_dim = self.hidden_dim * 2
        
        #  2. ATTENTION LAYERS (Bahdanau) 
        # e_ij = v_a * tanh(W_a * s_{i-1} + U_a * h_j)
        
        # W_a: Chiếu Decoder Hidden state (s_{i-1})
        self.W_a = nn.Linear(self.enc_output_dim, self.enc_output_dim) 
        
        # U_a: Chiếu Encoder Outputs (h_j)
        self.U_a = nn.Linear(self.enc_output_dim, self.enc_output_dim) 
        
        # v_a: Chiếu về 1 scalar score
        self.v_a = nn.Linear(self.enc_output_dim, 1, bias=False)
        
        #  3. DECODER (Multi-layer LSTMCells) 
        # Vì cần can thiệp vào từng bước time-step để tính Attention -> dùng ModuleList chứa các LSTMCell thay vì nn.LSTM.
        
        self.decoder_cells = nn.ModuleList()
        
        for i in range(self.decoder_layers):
            # Input của layer đầu tiên = Embedding + Context Vector
            if i == 0:
                input_size = self.embedding_dim + self.enc_output_dim
            else:
                # Input của các layer sau = Hidden state của layer trước (kích thước enc_output_dim)
                input_size = self.enc_output_dim
            
            self.decoder_cells.append(
                nn.LSTMCell(
                    input_size=input_size, 
                    hidden_size=self.enc_output_dim # Decoder hidden phải khớp với Encoder bidirectional
                )
            )

        # Output projection
        self.fc_out = nn.Linear(self.enc_output_dim, vocab.get_tgt_vocab_size())
        self.dropout = nn.Dropout(self.dropout_rate)

    def calculate_attention(self, s_prev, enc_outputs, mask=None):
        """
        s_prev: Hidden state của lớp Decoder CUỐI CÙNG (Top layer) tại bước trước [batch, dec_hidden_dim]
        enc_outputs: Output của encoder [batch, src_len, dec_hidden_dim]
        mask: Attention mask [batch, src_len] (1 = valid, 0 = padding)
        """
        # s_prev: [batch, dec_hidden] -> [batch, 1, dec_hidden]
        s_prev_expanded = s_prev.unsqueeze(1)
        
        # Energy: [batch, src_len, dec_hidden]
        energy = torch.tanh(self.W_a(s_prev_expanded) + self.U_a(enc_outputs))
        
        # Scores: [batch, src_len, 1]
        scores = self.v_a(energy)
        
        # Apply mask to scores (before softmax)
        if mask is not None:
            # mask: [batch, src_len] -> [batch, src_len, 1]
            mask = mask.unsqueeze(-1)
            scores = scores + (1 - mask) * (-1e9)  # Set padding scores to very negative
        
        # Weights: [batch, src_len, 1]
        weights = F.softmax(scores, dim=1)
        
        # Context: [batch, dec_hidden]
        context = torch.sum(weights * enc_outputs, dim=1)
        
        return context

    def forward(self, x, y=None):
        """
        x: [batch, src_len]
        y: [batch, tgt_len] (Optional). Nếu None -> Chế độ Predict
        """
        batch_size = x.shape[0]
        
        # Tính số lượng token khác PAD trong mỗi câu
        src_lens = torch.sum(x != self.vocab.pad_id, dim=1).cpu()
        
        # Đảo ngược câu nguồn, chỉ đảo phần từ thật, giữ nguyên padding ở cuối
        x_reversed = x.clone()
        for i, length in enumerate(src_lens):
            x_reversed[i, :length] = torch.flip(x[i, :length], dims=[0])
        #  ENCODER STEP 
        embedded_x = self.dropout(self.src_embedding(x_reversed))
        
        # Sử dụng Packed Sequence để bỏ qua tính toán trên các token PAD
        packed_embedded = pack_padded_sequence(embedded_x, 
            src_lens, 
            batch_first=True, 
            enforce_sorted=False
        )
        # enc_outputs: [batch, src_len, hidden_dim * 2]
        # hidden, cell: [2 * enc_layers, batch, hidden_dim]
        enc_outputs, (hidden, cell) = self.encoder(packed_embedded)
        
        #  PREPARE DECODER STATES
        # Reshape để tách chiều Forward/Backward
        # [2*layers, batch, hidden] -> [layers, 2, batch, hidden]
        hidden = hidden.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        
        # Concat Forward và Backward để khớp với decoder_hidden_dim (là 2*hidden_dim)
        # -> [layers, batch, 2*hidden]
        dec_hidden_states = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        dec_cell_states = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        
        # Chuyển thành list các tensor để dễ quản lý từng layer trong vòng lặp
        # Mỗi phần tử trong list là trạng thái của 1 layer: [batch, 2*hidden]
        h_states = [dec_hidden_states[i] for i in range(self.decoder_layers)]
        c_states = [dec_cell_states[i] for i in range(self.decoder_layers)]

        #  DECODER LOOP 
        outputs = []
        
        # Xác định độ dài vòng lặp
        if y is not None:
            max_len = y.shape[1]
            # Lấy input đầu tiên từ y (thường là <BOS>)
            decoder_input = y[:, 0] 
        else:
            max_len = self.config['max_length']
            # Tạo input <BOS>
            decoder_input = torch.tensor([self.vocab.bos_id] * batch_size, device=x.device)

        # Bắt đầu từ 1 vì 0 là <BOS> đã xử lý
        # Tuy nhiên nếu y=None, loop đến max_len. 
        # Nếu y!=None, loop đến max_len hoặc đến khi gặp EOS.      
       
        # Tạo attention mask từ src_lens: 1 = valid token, 0 = padding
        # mask shape: [batch, src_len]
        max_src_len = x.shape[1]
        attention_mask = torch.arange(max_src_len).unsqueeze(0).to(x.device) < src_lens.unsqueeze(1).to(x.device)
        attention_mask = attention_mask.float()  # [batch, src_len]
        num_steps = range(1, max_len) if y is not None else range(max_len)
        
        for t in num_steps:
            # Embed input: [batch, emb_dim]
            emb_input = self.dropout(self.tgt_embedding(decoder_input))
            
            # Calculate Attention
            # Dùng hidden state của layer cuối cùng (top layer) từ bước trước để tính attention
            s_top_prev = h_states[-1] 
            
            enc_outputs_unpacked, batch_size = pad_packed_sequence(enc_outputs, batch_first=True, total_length=x.shape[1])
            context = self.calculate_attention(s_top_prev, enc_outputs_unpacked, mask=attention_mask)
            
            # Decoder Layer 0 (Input + Context)
            # print(emb_input.shape, context.shape)
            emb_and_context = torch.cat((emb_input, context), dim=1) # [batch, emb_dim + enc_output_dim]
            
            h_states[0], c_states[0] = self.decoder_cells[0](emb_and_context, (h_states[0], c_states[0]))
            
            # Input cho các layer tiếp theo là hidden state của layer trước
            current_layer_input = h_states[0]
            
            # Decoder Layers 1...N (Duyệt xong layer 0 rồi duyệt tiếp các layer còn lại trong cùng 1 time-step)
            for i in range(1, self.decoder_layers):
                # Có thể thêm dropout giữa các layer LSTM nếu muốn
                # current_layer_input = self.dropout(current_layer_input) 
                h_states[i], c_states[i] = self.decoder_cells[i](current_layer_input, (h_states[i], c_states[i]))
                current_layer_input = h_states[i]
            
            # Prediction (Dùng output của layer cuối cùng)
            prediction = self.fc_out(h_states[-1]) # [batch, vocab_size]
            outputs.append(prediction.unsqueeze(1))
            
            # Chọn input cho bước tiếp theo
            if y is not None:
                # Training: Teacher Forcing (Dùng từ thật)
                decoder_input = y[:, t] 
            else:
                # Inference: Dùng từ vừa dự đoán
                decoder_input = prediction.argmax(1)
                
                # Early break nếu tất cả sample trong batch gặp EOS
                if (decoder_input == self.vocab.eos_id).all():
                    break

        # Nối các output lại: [batch, seq_len, vocab_size]
        logits = torch.cat(outputs, dim=1)
        
        return logits