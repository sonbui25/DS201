import torch
from torch import nn
import torch.nn.functional as F
import random

# Local Attention based on "Effective Approaches to Attention-based Neural Machine Translation" (Luong et al., 2015)
class LSTM_Local_Attention(nn.Module):
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
        # Lấy D từ config, nếu không có thì mặc định là 10 (theo bài báo)
        self.D = config.get('D', 10) 

        #  Embeddings 
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
        self.dropout = nn.Dropout(self.dropout_rate)

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
        self.dec_hidden_dim = self.enc_output_dim # Khớp kích thước

        #  2. LOCAL-P ATTENTION COMPONENTS (General Score) 
        
        # W_a: Dùng cho General Score (h_t^T * W_a * h_s)
        self.W_a = nn.Linear(self.enc_output_dim, self.dec_hidden_dim, bias=False)

        # W_p, v_p: Dùng để dự đoán vị trí p_t
        self.W_p = nn.Linear(self.dec_hidden_dim, self.dec_hidden_dim)
        self.v_p = nn.Linear(self.dec_hidden_dim, 1)

        # W_c: Dùng cho Input Feeding và tính Attentional Vector (h_tilde)
        self.W_c = nn.Linear(self.enc_output_dim + self.dec_hidden_dim, self.dec_hidden_dim)

        #  3. DECODER (Multi-layer LSTMCells) 
        self.decoder_cells = nn.ModuleList()
        
        for i in range(self.decoder_layers):
            # Input Layer 0: Embedding + Attentional Vector (Input Feeding)
            if i == 0:
                input_size = self.embedding_dim + self.dec_hidden_dim
            else:
                input_size = self.dec_hidden_dim
            
            self.decoder_cells.append(
                nn.LSTMCell(
                    input_size=input_size, 
                    hidden_size=self.dec_hidden_dim
                )
            )

        # Output projection
        self.fc_out = nn.Linear(self.dec_hidden_dim, vocab.get_tgt_vocab_size())
        
    def forward(self, x, y=None):
        """
        x: [batch, src_len]
        y: [batch, tgt_len] (Optional)
        """
        batch_size = x.shape[0]
        src_len = x.shape[1]

        #  0. REVERSE SOURCE 
        # Đảo ngược chuỗi nguồn (Source Sentence Reversing)
        x_reversed = torch.flip(x, dims=[1])
        
        #  ENCODER STEP 
        embedded_x = self.dropout(self.src_embedding(x_reversed))
        enc_outputs, (hidden, cell) = self.encoder(embedded_x)
        
        #  PREPARE DECODER STATES 
        # Khởi tạo hidden state cho Decoder từ trạng thái cuối của Encoder (Concat FWD/BWD)
        hidden = hidden.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        cell = cell.view(self.encoder_layers, 2, batch_size, self.hidden_dim)
        
        dec_hidden_states = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        dec_cell_states = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        
        h_states = [dec_hidden_states[i] for i in range(self.decoder_layers)]
        c_states = [dec_cell_states[i] for i in range(self.decoder_layers)]

        #  DECODER LOOP 
        outputs = []

        # Input Feeding: Khởi tạo attentional vector (h_tilde) ban đầu là 0
        attentional_hidden = torch.zeros(batch_size, self.dec_hidden_dim).to(x.device)

        if y is not None:
            max_len = y.shape[1]
            decoder_input = y[:, 0] 
        else:
            max_len = self.config['max_length']
            decoder_input = torch.tensor([self.vocab.bos_id] * batch_size, device=x.device)

        num_steps = range(1, max_len) if y is not None else range(max_len)
        
        for t in num_steps:
            # 1. Embed input
            emb_input = self.dropout(self.tgt_embedding(decoder_input))
            
            # 2. INPUT FEEDING: Concat Embedding + Previous Attentional Vector
            rnn_input = torch.cat((emb_input, attentional_hidden), dim=1)
            
            # 3. Chạy các layer LSTM
            h_states[0], c_states[0] = self.decoder_cells[0](rnn_input, (h_states[0], c_states[0]))
            curr_input = h_states[0]
            
            for i in range(1, self.decoder_layers):
                h_states[i], c_states[i] = self.decoder_cells[i](curr_input, (h_states[i], c_states[i]))
                curr_input = h_states[i]
            
            h_t = h_states[-1] # Top layer output

            #  4. LOCAL-P ATTENTION 
            
            # A. Dự đoán vị trí p_t (Predictive Alignment)
            p_t = src_len * torch.sigmoid(self.v_p(torch.tanh(self.W_p(h_t))))
            
            # B. Tính General Score: h_t^T * W_a * h_s
            enc_proj = self.W_a(enc_outputs)
            scores = torch.bmm(h_t.unsqueeze(1), enc_proj.transpose(1, 2)).squeeze(1)

            # C. Gaussian Window Masking
            indices = torch.arange(src_len, device=x.device).unsqueeze(0)
            sigma = self.D / 2
            gaussian_weights = torch.exp(-((indices - p_t)**2) / (2 * sigma**2))
            
            # Apply Attention: Softmax(scores) * Gaussian
            align_weights = F.softmax(scores, dim=1) * gaussian_weights
            align_weights = align_weights / (torch.sum(align_weights, dim=1, keepdim=True) + 1e-10)
            
            # BỎ: Lưu weights (saved_attentions.append...)

            # D. Context Vector: Sum(weights * enc_outputs)
            context = torch.bmm(align_weights.unsqueeze(1), enc_outputs).squeeze(1)

            #  5. ATTENTIONAL VECTOR (h_tilde) 
            # h_tilde = Tanh(W_c * [c_t; h_t]) -> Dùng cho Input Feeding bước sau
            concat_c_h = torch.cat((context, h_t), dim=1)
            attentional_hidden = torch.tanh(self.W_c(concat_c_h))
            
            #  6. PREDICTION 
            prediction = self.fc_out(attentional_hidden)
            outputs.append(prediction.unsqueeze(1))
            
            # Chọn input tiếp theo
            if y is not None:
                decoder_input = y[:, t] # Teacher Forcing
            else:
                decoder_input = prediction.argmax(1)
                if (decoder_input == self.vocab.eos_id).all():
                    break
        
        logits = torch.cat(outputs, dim=1)
                
        return logits
