import json
import re
import torch
from torch.utils.data import Dataset, DataLoader
from .tokenizer import Tokenizer

class NMTDataset(Dataset):
    """机器翻译数据集类"""
    def __init__(self, file_path, src_tokenizer, tgt_tokenizer, max_length=100, is_test=False):
        self.file_path = file_path
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.data = self._load_data()
    
    def _load_data(self):
        """加载JSONL数据"""
        data = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line.strip())
                    # 支持不同的字段名
                    if 'zh' in item and 'en' in item:
                        data.append((item['zh'], item['en']))
                    elif 'chinese' in item and 'english' in item:
                        data.append((item['chinese'], item['english']))
                    elif 'src' in item and 'tgt' in item:
                        data.append((item['src'], item['tgt']))
                except json.JSONDecodeError:
                    continue
        return data
    
    def _clean_text(self, text):
        """文本清洗"""
        # 移除非法字符和多余空格
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;]', '', text)
        text = ' '.join(text.split())
        return text.strip()
    
    def _truncate_sequence(self, ids):
        """截断过长序列"""
        if len(ids) > self.max_length:
            ids = ids[:self.max_length-1] + [ids[-1]]  # 保留最后一个标记
        return ids
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        
        # 文本清洗
        src_text = self._clean_text(src_text)
        tgt_text = self._clean_text(tgt_text)
        
        # 编码
        src_ids = self.src_tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)
        
        # 截断过长序列
        src_ids = self._truncate_sequence(src_ids)
        tgt_ids = self._truncate_sequence(tgt_ids)
        
        if self.is_test:
            return {
                'src_ids': torch.tensor(src_ids, dtype=torch.long),
                'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
                'src_text': src_text,
                'tgt_text': tgt_text
            }
        else:
            return {
                'src_ids': torch.tensor(src_ids, dtype=torch.long),
                'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long)
            }

def collate_fn(batch, pad_token_id=0):
    """批处理函数"""
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]
    
    # 填充序列
    src_max_len = max(len(ids) for ids in src_ids)
    tgt_max_len = max(len(ids) for ids in tgt_ids)
    
    src_padded = torch.full((len(batch), src_max_len), pad_token_id, dtype=torch.long)
    tgt_padded = torch.full((len(batch), tgt_max_len), pad_token_id, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_ids, tgt_ids)):
        src_padded[i, :len(src)] = src
        tgt_padded[i, :len(tgt)] = tgt
    
    result = {
        'src_ids': src_padded,
        'tgt_ids': tgt_padded,
        'src_lengths': torch.tensor([len(ids) for ids in src_ids], dtype=torch.long),
        'tgt_lengths': torch.tensor([len(ids) for ids in tgt_ids], dtype=torch.long)
    }
    
    if 'src_text' in batch[0]:
        result['src_texts'] = [item['src_text'] for item in batch]
        result['tgt_texts'] = [item['tgt_text'] for item in batch]
    
    return result

def create_data_loaders(config, src_tokenizer, tgt_tokenizer):
    """创建数据加载器"""
    # 兼容两种配置结构：顶层和data块
    if 'data' in config:
        data_config = config.get('data', {})
    else:
        data_config = config
    
    batch_size = data_config.get('batch_size', 32)
    max_length = data_config.get('max_length', 100)
    
    # 获取数据文件路径（支持两种配置结构）
    train_file = data_config.get('train_file') or config.get('train_file')
    val_file = data_config.get('val_file') or config.get('val_file')
    test_file = data_config.get('test_file') or config.get('test_file')
    
    if not train_file or not val_file or not test_file:
        raise ValueError("配置文件中缺少必要的训练、验证或测试文件路径")
    
    # 数据集
    train_dataset = NMTDataset(
        train_file, 
        src_tokenizer, 
        tgt_tokenizer, 
        max_length=max_length
    )
    
    val_dataset = NMTDataset(
        val_file, 
        src_tokenizer, 
        tgt_tokenizer, 
        max_length=max_length
    )
    
    test_dataset = NMTDataset(
        test_file, 
        src_tokenizer, 
        tgt_tokenizer, 
        max_length=max_length,
        is_test=True
    )
    
    # 数据加载器 - 优化配置
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, pad_token_id=0),
        num_workers=8,  # 增加工作进程数
        pin_memory=True,  # 启用内存锁定，加速GPU传输
        prefetch_factor=2,  # 预取因子
        persistent_workers=True  # 保持工作进程
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_token_id=0),
        num_workers=4,  
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, pad_token_id=0),
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader