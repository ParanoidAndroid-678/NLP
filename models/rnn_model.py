import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .attention import Attention

class RNNEncoder(nn.Module):
    """RNN编码器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, 
                 dropout=0.3, rnn_type='GRU'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 选择RNN类型
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False
            )
        else:
            raise ValueError(f"不支持的RNN类型: {rnn_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_lengths):
        """
        src: [seq_len, batch_size]
        src_lengths: [batch_size]
        """
        # 词嵌入
        embedded = self.dropout(self.embedding(src))  # [seq_len, batch_size, embed_size]
        
        # 打包序列
        packed_embedded = pack_padded_sequence(embedded, src_lengths, enforce_sorted=False)
        
        # 通过RNN
        if self.rnn_type == 'GRU':
            outputs, hidden = self.rnn(packed_embedded)
        else:  # LSTM
            outputs, (hidden, cell) = self.rnn(packed_embedded)
        
        # 解包序列
        outputs, _ = pad_packed_sequence(outputs)  # [seq_len, batch_size, hidden_size]
        
        if self.rnn_type == 'GRU':
            return outputs, hidden
        else:
            return outputs, (hidden, cell)

class RNNDecoder(nn.Module):
    """带注意力的RNN解码器"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2,
                 dropout=0.3, rnn_type='GRU', attention_method='dot'):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # 注意力机制
        self.attention = Attention(attention_method, hidden_size)
        
        # 选择RNN类型
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=embed_size + hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0
            )
        else:  # LSTM
            self.rnn = nn.LSTM(
                input_size=embed_size + hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0
            )
        
        # 输出层
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input, hidden, encoder_outputs):
        """
        input: [batch_size]
        hidden: 解码器的隐藏状态
        encoder_outputs: [seq_len, batch_size, hidden_size]
        """
        input = input.unsqueeze(0)  # [1, batch_size]
        
        # 词嵌入
        embedded = self.dropout(self.embedding(input))  # [1, batch_size, embed_size]
        
        # 获取上下文向量
        if self.rnn_type == 'GRU':
            hidden_for_attention = hidden[-1]  # 使用最后一层隐藏状态
        else:  # LSTM
            hidden_for_attention = hidden[0][-1]  # (h, c)中的h
        
        context, attention_weights = self.attention(hidden_for_attention, encoder_outputs)
        context = context.unsqueeze(0)  # [1, batch_size, hidden_size]
        
        # 连接嵌入和上下文
        rnn_input = torch.cat((embedded, context), dim=2)  # [1, batch_size, embed_size + hidden_size]
        
        # 通过RNN
        if self.rnn_type == 'GRU':
            output, hidden = self.rnn(rnn_input, hidden)
        else:  # LSTM
            output, (hidden, cell) = self.rnn(rnn_input, hidden)
            hidden = (hidden, cell)
        
        # 准备输出
        output = output.squeeze(0)  # [batch_size, hidden_size]
        context = context.squeeze(0)  # [batch_size, hidden_size]
        
        # 连接输出和上下文
        output = torch.cat((output, context), dim=1)  # [batch_size, hidden_size * 2]
        
        # 全连接层
        prediction = self.fc_out(output)  # [batch_size, vocab_size]
        
        return prediction, hidden, attention_weights

class RNNSeq2Seq(nn.Module):
    """完整的RNN序列到序列模型"""
    def __init__(self, encoder, decoder, device, teacher_forcing_ratio=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
    
    def forward(self, src, src_lengths, trg):
        """
        src: [src_seq_len, batch_size]
        src_lengths: [batch_size]
        trg: [trg_seq_len, batch_size]
        """
        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.vocab_size
        
        # 存储输出
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # 编码
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # 解码器的第一个输入是<sos>标记
        input = trg[0, :]  # 第一个时间步是<sos>
        
        # 解码循环
        for t in range(1, trg_len):
            # 解码
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs)
            
            # 存储输出
            outputs[t] = output
            
            # 决定下一个输入：Teacher forcing或贪婪选择
            teacher_force = torch.rand(1).item() < self.teacher_forcing_ratio
            
            # 获取最高概率的词
            top1 = output.argmax(1)
            
            # 更新输入
            input = trg[t] if teacher_force else top1
        
        return outputs
    
    def translate(self, src, src_lengths, max_len=50, beam_size=4):
        """
        翻译函数，支持贪心搜索和束搜索
        src: [src_seq_len, batch_size]
        src_lengths: [batch_size]
        """
        batch_size = src.shape[1]
        
        # 导入pad_sequence函数，确保两个分支都能使用
        from torch.nn.utils.rnn import pad_sequence
        
        with torch.no_grad():
            # 编码
            encoder_outputs, hidden = self.encoder(src, src_lengths)
            
            if beam_size > 1:
                # 为每个batch单独进行束搜索
                all_results = []
                for batch_idx in range(batch_size):
                    # 当前batch的编码器输出和隐藏状态
                    enc_out = encoder_outputs[:, batch_idx:batch_idx+1]  # [seq_len, 1, hidden_size]
                    if isinstance(hidden, tuple):  # LSTM
                        h_batch = (hidden[0][:, batch_idx:batch_idx+1], hidden[1][:, batch_idx:batch_idx+1])
                    else:  # GRU
                        h_batch = hidden[:, batch_idx:batch_idx+1]
                    
                    # 单个batch的束搜索
                    result = self._beam_search(enc_out, h_batch, max_len, beam_size)
                    all_results.append(torch.tensor(result, device=self.device))
                
                # 将所有batch的结果堆叠起来
                padded_results = pad_sequence(all_results, batch_first=True, padding_value=0)
                return padded_results
            else:
                # 贪心解码
                all_results = []
                for batch_idx in range(batch_size):
                    enc_out = encoder_outputs[:, batch_idx:batch_idx+1]
                    if isinstance(hidden, tuple):  # LSTM
                        h_batch = (hidden[0][:, batch_idx:batch_idx+1], hidden[1][:, batch_idx:batch_idx+1])
                    else:  # GRU
                        h_batch = hidden[:, batch_idx:batch_idx+1]
                    
                    result = self._greedy_decode(enc_out, h_batch, max_len)
                    all_results.append(torch.tensor(result, device=self.device))
                
                padded_results = pad_sequence(all_results, batch_first=True, padding_value=0)
                return padded_results

    def _greedy_decode(self, encoder_outputs, hidden, max_len):
        """贪心解码"""
        batch_size = encoder_outputs.shape[1]
        
        # 起始标记 - 修正：<sos>的ID是1，不是0
        input = torch.tensor([1]).to(self.device)  # <sos>的ID是1
        
        # 存储输出序列
        decoded_words = []
        
        for t in range(max_len):
            # 解码
            output, hidden, attention_weights = self.decoder(input, hidden, encoder_outputs)
            
            # 获取最高概率的词
            top1 = output.argmax(1).item()
            
            # 如果是<eos>则停止 - 修正：<eos>的ID是2，不是1
            if top1 == 2:  # <eos>的ID是2
                break
            
            decoded_words.append(top1)
            
            # 更新输入
            input = torch.tensor([top1]).to(self.device)
        
        return decoded_words
    
    def _beam_search(self, encoder_outputs, hidden, max_len, beam_size):
        """束搜索"""
        # 起始标记 - 修正：<sos>的ID是1，<eos>的ID是2
        start_token = 1  # <sos>的ID是1
        eos_token = 2    # <eos>的ID是2
        
        # 初始化束
        beams = [{
            'sequence': [start_token],
            'score': 0.0,
            'hidden': hidden
        }]
        
        for t in range(max_len):
            candidates = []
            
            for beam in beams:
                # 如果序列已经以<eos>结束，直接加入候选
                if beam['sequence'][-1] == eos_token:
                    candidates.append(beam)
                    continue
                
                # 解码
                input = torch.tensor([beam['sequence'][-1]]).to(self.device)
                output, new_hidden, _ = self.decoder(input, beam['hidden'], encoder_outputs)
                
                # 获取top-k个候选
                log_probs = F.log_softmax(output, dim=1)
                topk_probs, topk_indices = log_probs.topk(beam_size, dim=1)
                
                for i in range(beam_size):
                    new_score = beam['score'] + topk_probs[0, i].item()
                    new_sequence = beam['sequence'] + [topk_indices[0, i].item()]
                    
                    candidates.append({
                        'sequence': new_sequence,
                        'score': new_score,
                        'hidden': new_hidden
                    })
            
            # 选择得分最高的beam_size个序列
            candidates.sort(key=lambda x: x['score'] / len(x['sequence']), reverse=True)
            beams = candidates[:beam_size]
        
        # 返回最佳序列（去掉<sos>标记）
        best_sequence = beams[0]['sequence'][1:]
        
        # 如果序列以<eos>结束，去掉它
        if best_sequence and best_sequence[-1] == eos_token:
            best_sequence = best_sequence[:-1]
        
        return best_sequence