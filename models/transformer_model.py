import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from transformers import T5ForConditionalGeneration, T5Tokenizer

class PositionalEncoding(nn.Module):
    """绝对位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class RelativePositionalEncoding(nn.Module):
    """相对位置编码（简化版）"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(2 * max_len + 1, d_model)
    
    def forward(self, x):
        seq_len = x.size(0)
        batch_size = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        relative_positions = positions.unsqueeze(1) - positions.unsqueeze(0)
        relative_positions = torch.clamp(relative_positions, -self.max_len, self.max_len)
        relative_positions = relative_positions + self.max_len
        
        pos_encoding = self.embedding(relative_positions)  # [seq_len, seq_len, d_model]
        pos_encoding_mean = pos_encoding.mean(dim=1)  # [seq_len, d_model]
        pos_encoding_expanded = pos_encoding_mean.unsqueeze(1).expand(seq_len, batch_size, self.d_model)  # 修复：扩展batch维度
        return x + pos_encoding_expanded

class LayerNorm(nn.Module):
    """标准LayerNorm实现"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class RMSNorm(nn.Module):
    """RMSNorm实现"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * x / rms

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size, q_seq_len = q.size(0), q.size(1)
        k_seq_len = k.size(1)
        v_seq_len = v.size(1)
        
        # 线性变换并分头
        Q_linear = self.w_q(q)
        K_linear = self.w_k(k)
        V_linear = self.w_v(v)
        
        # 获取实际的特征维度
        actual_feature_dim = Q_linear.size(-1)
        
        # 确保张量连续
        Q_linear = Q_linear.contiguous()
        K_linear = K_linear.contiguous()
        V_linear = V_linear.contiguous()
        
        # 检查是否可以按预期分头
        if actual_feature_dim == self.d_model and actual_feature_dim % self.n_heads == 0:
            # 正常情况：使用预期的参数
            # Q使用q的序列长度，K使用k的序列长度，V使用v的序列长度
            Q = Q_linear.view(batch_size, q_seq_len, self.n_heads, self.d_k).transpose(1, 2)
            K = K_linear.view(batch_size, k_seq_len, self.n_heads, self.d_k).transpose(1, 2)
            V = V_linear.view(batch_size, v_seq_len, self.n_heads, self.d_k).transpose(1, 2)
        else:
            # 自适应分头：根据实际特征维度调整
            # 计算可以整除的最大头数
            max_heads = actual_feature_dim
            for h in range(self.n_heads, 0, -1):
                if actual_feature_dim % h == 0:
                    max_heads = h
                    break
            
            actual_heads = max_heads
            actual_d_k = actual_feature_dim // actual_heads
            
            # Q使用q的序列长度，K使用k的序列长度，V使用v的序列长度
            Q = Q_linear.view(batch_size, q_seq_len, actual_heads, actual_d_k).transpose(1, 2)
            K = K_linear.view(batch_size, k_seq_len, actual_heads, actual_d_k).transpose(1, 2)
            V = V_linear.view(batch_size, v_seq_len, actual_heads, actual_d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, q_seq_len, self.d_model
        )
        
        # 输出线性变换
        output = self.w_o(attn_output)
        
        return output, attn_weights

class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 选择归一化类型
        if norm_type == 'layernorm':
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_type='layernorm'):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        
        # 选择归一化类型
        if norm_type == 'layernorm':
            self.norm1 = LayerNorm(d_model)
            self.norm2 = LayerNorm(d_model)
            self.norm3 = LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            self.norm1 = RMSNorm(d_model)
            self.norm2 = RMSNorm(d_model)
            self.norm3 = RMSNorm(d_model)
        else:
            raise ValueError(f"不支持的归一化类型: {norm_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 自注意力子层（带因果mask）
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力子层
        cross_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_output))
        
        # 前馈子层
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, 
                 dropout=0.1, pos_encoding='absolute', norm_type='layernorm'):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        if pos_encoding == 'absolute':
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        elif pos_encoding == 'relative':
            self.pos_encoding = RelativePositionalEncoding(d_model, max_seq_len)
        else:
            raise ValueError(f"不支持的位置编码: {pos_encoding}")
        
        # 编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 选择归一化类型
        if norm_type == 'layernorm':
            self.norm = LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
    
    def forward(self, src, src_mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, src_mask)
        
        return self.norm(x)

class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len,
                 dropout=0.1, pos_encoding='absolute', norm_type='layernorm'):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        if pos_encoding == 'absolute':
            self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        elif pos_encoding == 'relative':
            self.pos_encoding = RelativePositionalEncoding(d_model, max_seq_len)
        else:
            raise ValueError(f"不支持的位置编码: {pos_encoding}")
        
        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout, norm_type)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # 输出层
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # 选择归一化类型
        if norm_type == 'layernorm':
            self.norm = LayerNorm(d_model)
        elif norm_type == 'rmsnorm':
            self.norm = RMSNorm(d_model)
    
    def forward(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        x = self.norm(x)
        
        # 输出投影
        output = self.output_projection(x)
        
        return output

class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_layers=6, 
                 n_heads=8, d_ff=2048, max_seq_len=5000, dropout=0.1,
                 pos_encoding='absolute', norm_type='layernorm'):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len,
            dropout, pos_encoding, norm_type
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len,
            dropout, pos_encoding, norm_type
        )
        
        # 参数初始化
        self._init_weights()
    
    def _init_weights(self):
        """参数初始化"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_src_mask(self, src):
        """创建源序列mask（用于padding）"""
        return (src != 0).unsqueeze(1).unsqueeze(2)
    
    def create_tgt_mask(self, tgt):
        """创建目标序列mask（因果mask + padding mask）"""
        seq_len = tgt.size(1)
        device = tgt.device  # 获取输入张量的设备
        
        # 因果mask（下三角矩阵）- 确保在正确的设备上，并转换为bool类型
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
        # padding mask
        padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        # 合并mask
        return causal_mask & padding_mask
    
    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
        return decoder_output
    
    def translate(self, src, max_len=50, beam_size=4):
        """翻译函数"""
        self.eval()
        
        with torch.no_grad():
            src_mask = self.create_src_mask(src)
            encoder_output = self.encoder(src, src_mask)
            
            if beam_size > 1:
                return self._beam_search(encoder_output, src_mask, max_len, beam_size)
            else:
                return self._greedy_decode(encoder_output, src_mask, max_len)
    
    def _greedy_decode(self, encoder_output, src_mask, max_len):
        """贪心解码 - 修复版本"""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # 起始标记
        tgt = torch.tensor([[1]] * batch_size, device=device)  # <sos>的ID是1
        
        # 记录每个序列是否已完成
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for i in range(max_len):
            tgt_mask = self.create_tgt_mask(tgt)
            output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
            
            # 应用softmax获取概率分布
            probs = F.softmax(output[:, -1], dim=-1)
            next_word = probs.argmax(dim=-1)
            
            # 添加到序列
            tgt = torch.cat([tgt, next_word.unsqueeze(1)], dim=1)
            
            # 更新完成状态
            finished = finished | (next_word == 2)  # <eos>的ID是2
            
            # 如果所有序列都已完成，则停止
            if finished.all():
                break
        
        # 调试信息：显示前几个序列的解码过程
        if batch_size > 0:
            print(f"解码结果 - 序列0: {tgt[0, :10].tolist()}...")
        
        return tgt[:, 1:]  # 去掉<sos>
    
    def _beam_search(self, encoder_output, src_mask, max_len, beam_size):
        """束搜索"""
        batch_size = encoder_output.size(0)
        device = encoder_output.device
        
        # 为每个batch单独进行束搜索
        all_results = []
        
        for batch_idx in range(batch_size):
            # 当前batch的编码器输出
            enc_out = encoder_output[batch_idx:batch_idx+1]  # [1, seq_len, d_model]
            src_mask_batch = src_mask[batch_idx:batch_idx+1] if src_mask is not None else None
            
            # 初始化束（每个batch单独维护）
            beams = [{
                'sequence': torch.tensor([1], device=device),  # <sos>的ID是1
                'score': 0.0,
                'length': 1
            }]
            
            for step in range(max_len):
                candidates = []
                
                for beam in beams:
                    # 如果序列以<eos>结束，直接保留
                    if beam['sequence'][-1] == 2:  # <eos>的ID是2
                        candidates.append(beam)
                        continue
                    
                    # 解码当前序列
                    tgt = beam['sequence'].unsqueeze(0)  # [1, seq_len]
                    tgt_mask = self.create_tgt_mask(tgt)
                    output = self.decoder(tgt, enc_out, src_mask_batch, tgt_mask)
                    
                    # 获取最后一个时间步的log概率
                    log_probs = F.log_softmax(output[0, -1], dim=-1)
                    topk_probs, topk_indices = log_probs.topk(beam_size)
                    
                    # 生成新候选
                    for i in range(beam_size):
                        new_seq = torch.cat([beam['sequence'], topk_indices[i].unsqueeze(0)])
                        # 使用长度归一化的分数（避免长序列得分过高）
                        new_length = beam['length'] + 1
                        new_score = (beam['score'] * beam['length'] + topk_probs[i].item()) / new_length
                        
                        candidates.append({
                            'sequence': new_seq,
                            'score': new_score,
                            'length': new_length
                        })
                
                # 选择得分最高的beam_size个候选
                candidates.sort(key=lambda x: x['score'], reverse=True)
                beams = candidates[:beam_size]
                
                # 如果所有beam都以<eos>结束，提前终止
                if all(beam['sequence'][-1] == 2 for beam in beams):
                    break
            
            # 选择最佳序列（去掉<sos>）
            best_sequence = beams[0]['sequence'][1:]
            # 去掉末尾的<eos>（如果存在）
            if best_sequence.numel() > 0 and best_sequence[-1] == 2:
                best_sequence = best_sequence[:-1]
            
            all_results.append(best_sequence)
        
        # 将所有batch的结果堆叠起来
        # 需要处理不同长度的序列，使用pad_sequence
        from torch.nn.utils.rnn import pad_sequence
        padded_results = pad_sequence(all_results, batch_first=True, padding_value=0)
        return padded_results  # [batch_size, max_seq_len]

class T5Transformer(nn.Module):
    """基于T5的Transformer模型（微调）"""
    def __init__(self, model_name='t5-base'):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    def forward(self, src, tgt):
        # T5使用相同的输入格式
        return self.model(input_ids=src, labels=tgt)
    
    def translate(self, src, max_len=50, beam_size=4):
        """使用T5进行翻译"""
        self.eval()
        
        with torch.no_grad():
            # T5自带的生成方法
            outputs = self.model.generate(
                src,
                max_length=max_len,
                num_beams=beam_size,
                early_stopping=True,
                decoder_start_token_id=self.tokenizer.pad_token_id
            )
            
            return outputs

def create_transformer_model(src_vocab_size, tgt_vocab_size, config):
    """创建Transformer模型的工厂函数"""
    if config.get('use_pretrained', False):
        return T5Transformer(config.get('model_name', 't5-base'))
    else:
        return Transformer(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config.get('d_model', 512),
            n_layers=config.get('n_layers', 6),
            n_heads=config.get('n_heads', 8),
            d_ff=config.get('d_ff', 2048),
            max_seq_len=config.get('max_seq_len', 5000),
            dropout=config.get('dropout', 0.1),
            pos_encoding=config.get('pos_encoding', 'absolute'),
            norm_type=config.get('norm_type', 'layernorm')
        )