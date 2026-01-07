import json
import re
from collections import Counter
import torch
import os
from .tokenizer import Tokenizer

class DataPreprocessor:
    """数据预处理类"""
    def __init__(self, config):
        self.config = config
        self.src_sentences = []
        self.tgt_sentences = []
    
    def load_and_clean_data(self, file_path, max_samples=None):
        """加载和清洗数据"""
        src_sentences = []
        tgt_sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                    
                try:
                    item = json.loads(line.strip())
                    
                    # 支持不同的字段名
                    if 'zh' in item and 'en' in item:
                        src_text = item['zh']
                        tgt_text = item['en']
                    elif 'chinese' in item and 'english' in item:
                        src_text = item['chinese']
                        tgt_text = item['english']
                    else:
                        continue
                    
                    # 清洗文本
                    src_clean = self.clean_text(src_text)
                    tgt_clean = self.clean_text(tgt_text)
                    
                    # 过滤空文本和过长句子
                    if (src_clean and tgt_clean and 
                        len(src_clean.split()) <= self.config.get('max_length', 100) and
                        len(tgt_clean.split()) <= self.config.get('max_length', 100)):
                        src_sentences.append(src_clean)
                        tgt_sentences.append(tgt_clean)
                        
                except json.JSONDecodeError:
                    continue
        
        return src_sentences, tgt_sentences
    
    def clean_text(self, text):
        """清洗单条文本"""
        if not text:
            return ""
        
        # 移除非法字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s.,!?;:()\-\']', '', text)
        # 标准化空格
        text = re.sub(r'\s+', ' ', text)
        # 移除首尾空格
        text = text.strip()
        
        return text
    
    
    def prepare_datasets(self):
        """准备所有数据集"""
        datasets = {}
        
        # 训练集（大）
        print("处理训练集（大）...")
        src_train_large, tgt_train_large = self.load_and_clean_data(
            self.config['train_large_file'], 
            max_samples=100000  # 100k样本
        )
        datasets['train_large'] = (src_train_large, tgt_train_large)
        
        
        # 验证集
        print("处理验证集...")
        src_val, tgt_val = self.load_and_clean_data(
            self.config['val_file'], 
            max_samples=500  # 500样本
        )
        datasets['val'] = (src_val, tgt_val)
        
        # 测试集
        print("处理测试集...")
        src_test, tgt_test = self.load_and_clean_data(
            self.config['test_file'], 
            max_samples=200  # 200样本
        )
        datasets['test'] = (src_test, tgt_test)
        
        return datasets
    
    def save_processed_data(self, datasets, output_dir):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, (src_sentences, tgt_sentences) in datasets.items():
            output_file = os.path.join(output_dir, f'{dataset_name}.jsonl')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for src, tgt in zip(src_sentences, tgt_sentences):
                    item = {
                        'src': src,
                        'tgt': tgt
                    }
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            print(f"保存 {dataset_name}: {len(src_sentences)} 个样本到 {output_file}")

def build_vocabularies(config, train_sentences):
    """构建词汇表"""
    src_sentences, tgt_sentences = train_sentences
    
    # 中文分词器
    src_tokenizer = Tokenizer(language='chinese')
    src_tokenizer.build_vocab(
        src_sentences,
        max_vocab_size=config.get('max_vocab_size', 30000),
        min_freq=config.get('min_freq', 2)
    )
    
    # 英文分词器
    tgt_tokenizer = Tokenizer(language='english')
    tgt_tokenizer.build_vocab(
        tgt_sentences,
        max_vocab_size=config.get('max_vocab_size', 30000),
        min_freq=config.get('min_freq', 2)
    )
    
    print(f"中文词汇表大小: {src_tokenizer.vocab_size}")
    print(f"英文词汇表大小: {tgt_tokenizer.vocab_size}")
    
    return src_tokenizer, tgt_tokenizer

# 主处理函数
def preprocess_data(config):
    """数据预处理主函数"""
    print("开始数据预处理...")
    
    # 初始化预处理器
    preprocessor = DataPreprocessor(config)
    
    # 准备数据集
    datasets = preprocessor.prepare_datasets()
    
    # 保存处理后的数据
    output_dir = config.get('output_dir', './data/processed')
    preprocessor.save_processed_data(datasets, output_dir)
    
    # 构建词汇表（使用训练集）
    src_tokenizer, tgt_tokenizer = build_vocabularies(config, datasets['train_large'])
    
    # 更新配置文件
    config.update({
        'src_vocab_size': src_tokenizer.vocab_size,
        'tgt_vocab_size': tgt_tokenizer.vocab_size,
        'train_file': f'{output_dir}/train_large.jsonl',
        'train_small_file': f'{output_dir}/train_small.jsonl',
        'val_file': f'{output_dir}/val.jsonl',
        'test_file': f'{output_dir}/test.jsonl'
    })
    
    print("数据预处理完成！")
    return src_tokenizer, tgt_tokenizer, config