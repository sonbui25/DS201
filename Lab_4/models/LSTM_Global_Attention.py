import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Luong Global Attention based on "Effective Approaches to Attention-based Neural Machine Translation" (Luong et al., 2015)
class LSTM_Global_Attention(nn.Module):
    def __init__(self, vocab, config):
        super().__init__()
        self.config = config
        self.vocab = vocab
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dimensions
        self.embedding_dim = config['embedding_dim']
        self.hidden_dim = config['hidden_dim']
        self.encoder_layers = config['encoder_layers']
        self.decoder_layers = config['decoder_layers']
        self.dropout_rate = config['dropout']
        
        # Encoder bidirectional -> Decoder hidden_dim = 2 * encoder_hidden_dim
        self.dec_hidden_dim = self.hidden_dim * 2

        # Embedding
        self.src_embedding = nn.Embedding(vocab.get_src_vocab_size(), self.embedding_dim, padding_idx=vocab.pad_id)
        self.tgt_embedding = nn.Embedding(vocab.get_tgt_vocab_size(), self.embedding_dim, padding_idx=vocab.pad_id)

        # 1. ENCODER (Bidirectional)
        self.encoder = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.encoder_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.encoder_layers > 1 else 0
        )
        
        # 2. ATTENTION LAYER (Luong - General Score)
        # score(h_t, h_s) = h_t^T * W_a * h_s
        self.W_a = nn.Linear(self.dec_hidden_dim, self.dec_hidden_dim, bias=False)
        
        # 3. ATTENTIONAL HIDDEN STATE LAYER (W_c)
        # h_tilde_t = tanh(W_c * [c_t; h_t])
        self.W_c = nn.Linear(self.dec_hidden_dim * 2, self.dec_hidden_dim, bias=False)

        # 4. DECODER CELLS (Luong style with Input-feeding)
        self.decoder_cells = nn.ModuleList()
        for i in range(self.decoder_layers):
            # Input-feeding: input = Embedding + h_tilde_{t-1} (ở layer 0)
            input_size = (self.embedding_dim + self.dec_hidden_dim) if i == 0 else self.dec_hidden_dim
            self.decoder_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=self.dec_hidden_dim))

        # Output projection (Từ h_tilde_t dự đoán từ tiếp theo)
        self.fc_out = nn.Linear(self.dec_hidden_dim, vocab.get_tgt_vocab_size())
        self.dropout = nn.Dropout(self.dropout_rate)

    def calculate_attention(self, h_t, enc_outputs, mask=None):
        """
        h_t: Hidden state HIỆN TẠI của top decoder layer [batch, dec_hidden_dim]
        enc_outputs: [batch, src_len, dec_hidden_dim]
        """
        # score = h_t * W_a * enc_outputs
        # [batch, 1, dec_hidden] * [batch, dec_hidden, src_len] -> [batch, 1, src_len]
        wa_h_t = self.W_a(h_t).unsqueeze(1) 
        scores = torch.bmm(wa_h_t, enc_outputs.transpose(1, 2))
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        
        weights = F.softmax(scores, dim=-1) # [batch, 1, src_len]
        
        # Context vector c_t: Weighted sum of encoder outputs
        context = torch.bmm(weights, enc_outputs).squeeze(1) # [batch, dec_hidden_dim]
        
        return context

    def forward(self, x, y=None):
        """
        x: [batch, src_len]
        y: [batch, tgt_len] (Optional)
        Nếu y có thì dùng để Teacher Forcing, không thì dùng để Inference
        """
        batch_size = x.shape[0]
        src_lens = torch.sum(x != self.vocab.pad_id, dim=1).cpu()

        # Đảo ngược chuỗi đầu vào
        x_reversed = x.clone()
        for i, length in enumerate(src_lens):
            x_reversed[i, :length] = torch.flip(x_reversed[i, :length], dims=[0])
            
        #  ENCODER 
        embedded_x = self.dropout(self.src_embedding(x_reversed))
        
        packed_embedded = pack_padded_sequence(embedded_x, src_lens, batch_first=True, enforce_sorted=False)
        enc_outputs, (hidden, cell) = self.encoder(packed_embedded)
        enc_outputs_unpacked, _ = pad_packed_sequence(enc_outputs, batch_first=True, total_length=x.shape[1])
        
        # Reshape encoder states cho Decoder
        hidden = hidden.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        h_states = [torch.cat((hidden[i, 0], hidden[i, 1]), dim=-1) for i in range(self.decoder_layers)]
        c_states = [torch.cat((cell[i, 0], cell[i, 1]), dim=-1) for i in range(self.decoder_layers)]

        #  DECODER LOOP 
        max_len = y.shape[1] if y is not None else self.config['max_length']
        decoder_input = y[:, 0] if y is not None else torch.tensor([self.vocab.bos_id] * batch_size, device=self.device)
        
        # Khởi tạo h_tilde (attentional hidden state) ban đầu bằng 0 (cho bước input-feeding đầu tiên)
        h_tilde = torch.zeros(batch_size, self.dec_hidden_dim, device=self.device)
        
        attention_mask = torch.arange(x.shape[1]).unsqueeze(0).to(self.device) < src_lens.unsqueeze(1).to(self.device)
        outputs = []
        num_steps = range(1, max_len) if y is not None else range(max_len)

        for t in num_steps:
            # 1. Embedding + Input Feeding (Kết hợp embedding từ với h_tilde bước trước)
            emb_input = self.dropout(self.tgt_embedding(decoder_input))
            decoder_layer_input = torch.cat((emb_input, h_tilde), dim=1) 
            
            # 2. Cập nhật các lớp LSTM (Tính h_t trước)
            h_states[0], c_states[0] = self.decoder_cells[0](decoder_layer_input, (h_states[0], c_states[0]))
            for i in range(1, self.decoder_layers):
                h_states[i], c_states[i] = self.decoder_cells[i](h_states[i-1], (h_states[i], c_states[i]))
            
            h_t = h_states[-1] # Hidden state hiện tại của top layer

            # 3. Tính Global Attention dựa trên h_t hiện tại
            context = self.calculate_attention(h_t, enc_outputs_unpacked, mask=attention_mask)
            
            # 4. Tạo Attentional Hidden State (h_tilde_t)
            # h_tilde = tanh(W_c[c_t; h_t])
            combined = torch.cat((context, h_t), dim=1)
            h_tilde = torch.tanh(self.W_c(combined))
            
            # 5. Prediction từ h_tilde
            prediction = self.fc_out(h_tilde)
            outputs.append(prediction.unsqueeze(1))
            
            # Chuẩn bị cho bước tiếp theo
            if y is not None:
                decoder_input = y[:, t]
            else:
                decoder_input = prediction.argmax(1)
                if (decoder_input == self.vocab.eos_id).all(): break
                
        # Nối các output lại: [batch, seq_len, vocab_size]
        logits = torch.cat(outputs, dim=1)
        return logits