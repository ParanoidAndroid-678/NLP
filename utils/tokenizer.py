import jieba
import re
from collections import Counter
import sentencepiece as spm
import os
import logging
jieba.setLogLevel(logging.WARNING)
class BaseTokenizer:
    """基础分词器类"""
    def __init__(self):
        self.vocab = None
        self.vocab_size = 0
        self.special_tokens = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    
    def build_vocab(self, sentences, max_vocab_size=30000, min_freq=2):
        """构建词汇表"""
        raise NotImplementedError
    
    def encode(self, text):
        """将文本编码为ID序列"""
        raise NotImplementedError
    
    def decode(self, ids):
        """将ID序列解码为文本"""
        raise NotImplementedError

class ChineseTokenizer(BaseTokenizer):
    """中文分词器（基于jieba）"""
    def __init__(self):
        super().__init__()
        jieba.initialize()
    
    def tokenize(self, text):
        """中文分词"""
        # 清洗文本
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', '', text)
        tokens = jieba.lcut(text)
        return [token for token in tokens if token.strip()]
    
    def build_vocab(self, sentences, max_vocab_size=30000, min_freq=2):
        """构建中文词汇表"""
        word_freq = Counter()
        
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            word_freq.update(tokens)
        
        vocab_words = ['<pad>', '<sos>', '<eos>', '<unk>']
        vocab_words.extend([word for word, freq in word_freq.most_common(max_vocab_size) 
                          if freq >= min_freq])
        
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        self.vocab_size = len(vocab_words)
        return self.vocab
    
    def encode(self, text, add_special_tokens=True):
        """编码中文文本"""
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.special_tokens['<unk>']) for token in tokens]
        
        if add_special_tokens:
            ids = [self.special_tokens['<sos>']] + ids + [self.special_tokens['<eos>']]
        
        return ids
    
    def decode(self, ids, remove_special_tokens=True):
        """解码中文文本"""
        if remove_special_tokens:
            ids = [id for id in ids if id not in [0, 1, 2, 3]]
        
        reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        tokens = [reverse_vocab.get(id, '<unk>') for id in ids]
        return ''.join(tokens)

class EnglishTokenizer(BaseTokenizer):
    """英文分词器"""
    def __init__(self):
        super().__init__()
    
    def tokenize(self, text):
        """英文分词"""
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        tokens = text.split()
        return [token for token in tokens if token.strip()]
    
    def build_vocab(self, sentences, max_vocab_size=30000, min_freq=2):
        """构建英文词汇表"""
        word_freq = Counter()
        
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            word_freq.update(tokens)
        
        vocab_words = ['<pad>', '<sos>', '<eos>', '<unk>']
        vocab_words.extend([word for word, freq in word_freq.most_common(max_vocab_size) 
                          if freq >= min_freq])
        
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        self.vocab_size = len(vocab_words)
        return self.vocab
    
    def encode(self, text, add_special_tokens=True):
        """编码英文文本"""
        tokens = self.tokenize(text)
        ids = [self.vocab.get(token, self.special_tokens['<unk>']) for token in tokens]
        
        if add_special_tokens:
            ids = [self.special_tokens['<sos>']] + ids + [self.special_tokens['<eos>']]
        
        return ids
    
    def decode(self, ids, remove_special_tokens=True):
        """解码英文文本"""
        if remove_special_tokens:
            ids = [id for id in ids if id not in [0, 1, 2, 3]]
        
        reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        tokens = [reverse_vocab.get(id, '<unk>') for id in ids]
        return ' '.join(tokens)

class Tokenizer:
    """统一分词器"""
    def __init__(self, language='chinese'):
        self.language = language
        
        if language == 'chinese':
            self.tokenizer = ChineseTokenizer()
        elif language == 'english':
            self.tokenizer = EnglishTokenizer()
        else:
            raise ValueError(f"不支持的语言: {language}")
    
    def build_vocab(self, sentences, **kwargs):
        return self.tokenizer.build_vocab(sentences, **kwargs)
    
    def encode(self, text, **kwargs):
        return self.tokenizer.encode(text, **kwargs)
    
    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, **kwargs)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    @property
    def vocab(self):
        return self.tokenizer.vocab