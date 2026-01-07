import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DotAttention(nn.Module):
    """点积注意力"""
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: [batch_size, hidden_size]
        encoder_outputs: [seq_len, batch_size, hidden_size]
        """
        # 计算注意力分数
        decoder_hidden = decoder_hidden.unsqueeze(1)  # [batch_size, 1, hidden_size]
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        # 点积注意力
        scores = torch.bmm(decoder_hidden, encoder_outputs.transpose(1, 2))
        scores = scores.squeeze(1)  # [batch_size, seq_len]
        
        # 计算注意力权重
        weights = F.softmax(scores, dim=1)
        
        # 计算上下文向量
        weights = weights.unsqueeze(2)  # [batch_size, seq_len, 1]
        context = torch.bmm(encoder_outputs.transpose(1, 2), weights)  # 移除多余的unsqueeze(-1)
        context = context.squeeze(-1)  # [batch_size, hidden_size]
        
        return context, weights.squeeze(2)

class MultiplicativeAttention(nn.Module):
    """乘性注意力（一般化点积）"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: [batch_size, hidden_size]
        encoder_outputs: [seq_len, batch_size, hidden_size]
        """
        # 变换decoder隐藏状态
        decoder_transformed = self.W(decoder_hidden)  # [batch_size, hidden_size]
        decoder_transformed = decoder_transformed.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        # 计算注意力分数
        scores = torch.bmm(decoder_transformed, encoder_outputs.transpose(1, 2))
        scores = scores.squeeze(1)  # [batch_size, seq_len]
        
        # 计算注意力权重
        weights = F.softmax(scores, dim=1)
        
        # 计算上下文向量
        weights = weights.unsqueeze(2)  # [batch_size, seq_len, 1]
        context = torch.bmm(encoder_outputs.transpose(1, 2), weights)  # 修复：移除多余的unsqueeze(-1)
        context = context.squeeze(-1)  # [batch_size, hidden_size]
        
        return context, weights.squeeze(2)

class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, hidden_size):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, decoder_hidden, encoder_outputs):
        """
        decoder_hidden: [batch_size, hidden_size]
        encoder_outputs: [seq_len, batch_size, hidden_size]
        """
        seq_len = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        
        # 变换decoder隐藏状态
        decoder_transformed = self.W1(decoder_hidden)  # [batch_size, hidden_size]
        decoder_transformed = decoder_transformed.unsqueeze(0).repeat(seq_len, 1, 1)  # [seq_len, batch_size, hidden_size]
        
        # 变换encoder输出
        encoder_transformed = self.W2(encoder_outputs)  # [seq_len, batch_size, hidden_size]
        
        # 计算加性注意力分数
        combined = torch.tanh(decoder_transformed + encoder_transformed)  # [seq_len, batch_size, hidden_size]
        scores = self.v(combined).squeeze(-1)  # [seq_len, batch_size]
        
        # 转置以匹配期望的形状
        scores = scores.transpose(0, 1)  # [batch_size, seq_len]
        
        # 计算注意力权重
        weights = F.softmax(scores, dim=1)
        
        # 计算上下文向量
        weights = weights.unsqueeze(2)  # [batch_size, seq_len, 1]
        encoder_outputs_batch_first = encoder_outputs.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        context = torch.bmm(encoder_outputs_batch_first.transpose(1, 2), weights)  # 修复：先转置为batch_first格式
        context = context.squeeze(-1)  # [batch_size, hidden_size]
        
        return context, weights.squeeze(2)

class Attention(nn.Module):
    """统一注意力模块，支持三种注意力机制"""
    def __init__(self, method, hidden_size):
        super().__init__()
        self.method = method
        
        if method == 'dot':
            self.attention = DotAttention(hidden_size)
        elif method == 'multiplicative':
            self.attention = MultiplicativeAttention(hidden_size)
        elif method == 'additive':
            self.attention = AdditiveAttention(hidden_size)
        else:
            raise ValueError(f"不支持的注意力机制: {method}")
    
    def forward(self, decoder_hidden, encoder_outputs):
        return self.attention(decoder_hidden, encoder_outputs)