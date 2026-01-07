import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import json
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt  # 添加matplotlib用于绘制loss曲线

from models.rnn_model import RNNSeq2Seq
from utils.data_loader import create_data_loaders
from utils.metrics import evaluate_translation_fast, evaluate_translation_standard, print_evaluation_results
from utils.tokenizer import Tokenizer

class RNNTrainer:
    """RNN模型训练器"""
    
    def __init__(self, config, src_tokenizer, tgt_tokenizer, device):
        self.config = config
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        
        # 创建模型
        self.model = self._create_model()
        self.model.to(device)
        
        # 优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_bleu_scores = []
        self.best_bleu = 0
        
        # 创建输出目录
        os.makedirs(config.get('output_dir', './checkpoints/rnn'), exist_ok=True)
    
    def _create_model(self):
        """创建RNN模型"""
        model_config = self.config['model']
        
        # 编码器
        from models.rnn_model import RNNEncoder, RNNDecoder
        
        encoder = RNNEncoder(
            vocab_size=self.src_tokenizer.vocab_size,
            embed_size=model_config.get('embed_size', 256),
            hidden_size=model_config.get('hidden_size', 512),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.3),
            rnn_type=model_config.get('rnn_type', 'GRU')  # GRU 或 LSTM
        )
        
        decoder = RNNDecoder(
            vocab_size=self.tgt_tokenizer.vocab_size,
            embed_size=model_config.get('embed_size', 256),
            hidden_size=model_config.get('hidden_size', 512),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.3),
            rnn_type=model_config.get('rnn_type', 'GRU'),
            attention_method=model_config.get('attention_method', 'dot')  # dot/multiplicative/additive
        )
        
        model = RNNSeq2Seq(
            encoder=encoder,
            decoder=decoder,
            device=self.device,
            teacher_forcing_ratio=model_config.get('teacher_forcing_ratio', 0.5)
        )
        
        print(f"创建RNN模型: {model_config.get('rnn_type')}")
        print(f"注意力机制: {model_config.get('attention_method')}")
        print(f"Teacher Forcing比率: {model_config.get('teacher_forcing_ratio', 0.5)}")
        
        return model
    
    def _create_optimizer(self):
        """创建优化器"""
        training_config = self.config['training']
        optimizer_type = training_config.get('optimizer', 'adam')
        lr = training_config.get('learning_rate', 0.001)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")
        
        return optimizer
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src_ids'].to(self.device)
            tgt = batch['tgt_ids'].to(self.device)
            src_lengths = batch['src_lengths']  # 保持CPU张量，不要移动到GPU
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src.t(), src_lengths, tgt.t())
            
            # 修复损失计算：确保维度匹配和时间步对齐
            # output形状: [trg_len, batch_size, vocab_size]
            # tgt.t()形状: [trg_len, batch_size]
            trg_len = output.shape[0]
            
            # 从第1个时间步开始，因为第0个时间步是空的
            # output[1:] 预测的是 tgt.t()[1:]
            output_dim = output.shape[-1]
            output_flat = output[1:].reshape(-1, output_dim)
            tgt_flat = tgt.t()[1:].reshape(-1)
            
            # 确保维度匹配 - 添加维度检查
            if output_flat.shape[0] != tgt_flat.shape[0]:
                # 如果维度不匹配，取较小的维度
                min_length = min(output_flat.shape[0], tgt_flat.shape[0])
                output_flat = output_flat[:min_length]
                tgt_flat = tgt_flat[:min_length]
                print(f"警告: 损失计算维度不匹配，已调整为 {min_length}")
            
            loss = self.criterion(output_flat, tgt_flat)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader, epoch, use_fast_eval=True):
        """验证模型 - 优化版本"""
        self.model.eval()
        total_loss = 0
        all_references = []
        all_hypotheses = []
        
        # 限制验证样本数量：训练过程中用200个样本，训练完成后用500个样本
        max_bleu_samples = 200 if use_fast_eval else 500  # 快速评估用200，标准评估用500
        processed_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} Validation'):
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                src_lengths = batch['src_lengths']  # 获取序列长度
                
                # 计算验证损失 - 使用相同的修复逻辑
                output = self.model(src.t(), src_lengths, tgt.t())
                output_dim = output.shape[-1]
                output_flat = output[1:].reshape(-1, output_dim)
                tgt_flat = tgt.t()[1:].reshape(-1)
                
                # 确保维度匹配
                if output_flat.shape[0] != tgt_flat.shape[0]:
                    min_length = min(output_flat.shape[0], tgt_flat.shape[0])
                    output_flat = output_flat[:min_length]
                    tgt_flat = tgt_flat[:min_length]
                    print(f"验证警告: 损失计算维度不匹配，已调整为 {min_length}")
                
                loss = self.criterion(output_flat, tgt_flat)
                total_loss += loss.item()
                
                # 只在需要时生成翻译用于BLEU计算
                if processed_samples < max_bleu_samples:
                    # 使用贪心解码加速
                    for i in range(min(src.size(0), max_bleu_samples - processed_samples)):
                        src_sentence = src[i:i+1].t()  # [seq_len, 1]
                        src_len = src_lengths[i:i+1]  # 获取序列长度
                        
                        # 使用贪心解码（beam_size=1）加速
                        translation_ids = self.model.translate(
                            src_sentence, 
                            src_len,
                            beam_size=1  # 改为贪心解码
                        )
                        
                        # 解码为文本
                        ref_text = self.tgt_tokenizer.decode(
                            tgt[i].cpu().numpy(), 
                            remove_special_tokens=True
                        )
                        hyp_text = self.tgt_tokenizer.decode(
                            translation_ids[0].cpu().numpy(),  # 添加索引[0]
                            remove_special_tokens=True
                        )
                        
                        all_references.append(ref_text)
                        all_hypotheses.append(hyp_text)
                        processed_samples += 1
        
        avg_loss = total_loss / len(val_loader)
        
        # 总是使用sacreBLEU计算BLEU-4，使用固定的语言参数
        from utils.metrics import evaluate_translation_fast as evaluate_func
        
        # 只在有样本时计算BLEU
        if all_references:
            eval_results = evaluate_func(
                all_references, 
                all_hypotheses, 
                'chinese',  # 固定源语言为中文
                'english'   # 固定目标语言为英文
            )
        else:
            eval_results = {'bleu': 0.0}
        
        # 添加评估类型信息
        eval_results['eval_type'] = "fast" if use_fast_eval else "standard"
        eval_results['sample_count'] = len(all_references)
        
        return avg_loss, eval_results
    
    def train(self, train_loader, val_loader):
        """完整训练过程 - 优化版本"""
        training_config = self.config['training']
        num_epochs = training_config.get('num_epochs', 30)
        patience = training_config.get('patience', 5)
        
        print("开始训练RNN模型...")
        print(f"训练周期: {num_epochs}")
        print(f"训练过程中BLEU评估: 每个epoch使用200个样本")
        print(f"训练完成后BLEU评估: 使用500个样本")
        print(f"设备: {self.device}")
        print("-" * 50)
        
        no_improvement = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 训练过程中总是使用快速评估（200样本）
            use_fast_eval = (epoch != num_epochs)
            
            # 验证
            val_loss, val_metrics = self.validate(val_loader, epoch, use_fast_eval=use_fast_eval)
            current_bleu = val_metrics['bleu']
            
            # 记录结果
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_bleu_scores.append(current_bleu)
            
            epoch_time = time.time() - start_time
            
            # 打印结果
            eval_type = "快速(200样本)" if use_fast_eval else "标准(500样本)"
            print(f"\nEpoch {epoch} 完成 - 时间: {epoch_time:.2f}s ({eval_type}评估)")
            print(f"训练损失: {train_loss:.4f}")
            print(f"验证损失: {val_loss:.4f}")
            print(f"验证BLEU: {current_bleu:.4f}")
            print(f"评估样本数: {val_metrics['sample_count']}")
            
            # 检查模型改进（基于快速评估）
            if current_bleu > self.best_bleu:
                self.best_bleu = current_bleu
                self.save_checkpoint(epoch, is_best=True)
                no_improvement = 0
                print(f"新的最佳模型! BLEU: {current_bleu:.4f}")
            else:
                no_improvement += 1
                print(f"模型未改进，跳过checkpoint保存")
            
            # 早停检查（基于快速评估）
            if no_improvement >= patience:
                print(f"早停: {patience}个周期没有改善")
                break
        
        # 训练完成后进行最终标准评估（1000样本）
        print("\n训练完成，进行最终标准评估...")
        final_val_loss, final_metrics = self.validate(val_loader, num_epochs, use_fast_eval=False)
        final_bleu = final_metrics['bleu']
        
        print(f"最终标准评估结果:")
        print(f"验证损失: {final_val_loss:.4f}")
        print(f"验证BLEU: {final_bleu:.4f}")
        print(f"评估样本数: {final_metrics['sample_count']}")
        
        # 如果最终评估的BLEU更好，更新最佳模型
        if final_bleu > self.best_bleu:
            self.best_bleu = final_bleu
            self.save_checkpoint(num_epochs, is_best=True)
            print(f"最终模型为最佳模型! BLEU: {final_bleu:.4f}")
        
        print(f"最佳BLEU分数: {self.best_bleu:.4f}")
        
        # 保存训练记录和绘制loss曲线
        self.save_training_history()
        self.plot_loss_curves()
        
        # 训练结束时保存最终模型（如果最终模型不是最佳模型）
        final_checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'val_bleu': self.val_bleu_scores[-1] if self.val_bleu_scores else 0,
            'best_bleu': self.best_bleu,
            'config': self.config
        }
        
        checkpoint_dir = self.config.get('output_dir', './checkpoints/rnn')
        final_path = os.path.join(checkpoint_dir, 'final_model.pth')
        torch.save(final_checkpoint, final_path)
        print(f"保存最终模型: epoch {epoch}")

    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点 - 只保存最佳模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'val_bleu': self.val_bleu_scores[-1] if self.val_bleu_scores else 0,
            'best_bleu': self.best_bleu,
            'config': self.config
        }
        
        checkpoint_dir = self.config.get('output_dir', './checkpoints/rnn')
        
        # 只保存最佳模型，不保存每个epoch的checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: epoch {epoch}, BLEU: {self.best_bleu:.4f}")
    
    def save_training_history(self):
        """保存训练历史"""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_bleu_scores': self.val_bleu_scores,
            'best_bleu': self.best_bleu,
            'config': self.config
        }
        
        history_path = os.path.join(
            self.config.get('output_dir', './checkpoints/rnn'),
            'training_history.json'
        )
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        
        print(f"训练记录已保存到: {history_path}")
    
    def plot_loss_curves(self):
        """绘制loss曲线"""
        if not self.train_losses or not self.val_losses:
            print("没有训练数据，无法绘制loss曲线")
            return
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 绘制loss曲线
        ax1.plot(epochs, self.train_losses, 'b-', label='train_loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='val_loss')
        ax1.set_title('loss curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制BLEU分数曲线
        if self.val_bleu_scores:
            ax2.plot(epochs, self.val_bleu_scores, 'g-', label='BLEU_score')
            ax2.set_title('BLEU curve')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('BLEU Score')
            ax2.legend()
            ax2.grid(True)
        
        # 保存图像
        plot_path = os.path.join(
            self.config.get('output_dir', './checkpoints/rnn'),
            'training_curves.png'
        )
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"loss曲线已保存到: {plot_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复训练状态
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_bleu_scores = checkpoint.get('val_bleu_scores', [])
        self.best_bleu = checkpoint.get('best_bleu', 0)
        
        print(f"加载检查点: {checkpoint_path}")
        print(f"训练周期: {checkpoint['epoch']}")
        print(f"最佳BLEU: {self.best_bleu:.4f}")

def compare_attention_mechanisms(config, src_tokenizer, tgt_tokenizer, device):
    """比较不同注意力机制"""
    attention_methods = ['dot', 'multiplicative', 'additive']
    results = {}
    
    for method in attention_methods:
        print(f"\n训练 {method} 注意力机制...")
        print("=" * 50)
        
        # 更新配置
        config['model']['attention_method'] = method
        config['output_dir'] = f"./checkpoints/rnn_{method}"
        
        # 创建训练器
        trainer = RNNTrainer(config, src_tokenizer, tgt_tokenizer, device)
        
        # 获取数据加载器
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 训练
        trainer.train(train_loader, val_loader)
        
        # 记录结果
        results[method] = {
            'best_bleu': trainer.best_bleu,
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else 0,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0
        }
    
    # 打印比较结果
    print("\n注意力机制比较结果:")
    print("=" * 50)
    for method, result in results.items():
        print(f"{method}: BLEU={result['best_bleu']:.4f}, "
              f"Val Loss={result['final_val_loss']:.4f}")

def compare_training_strategies(config, src_tokenizer, tgt_tokenizer, device):
    """比较不同训练策略"""
    strategies = [0.0, 0.5, 1.0]  # Teacher Forcing比率
    results = {}
    
    for ratio in strategies:
        print(f"\n训练 Teacher Forcing比率: {ratio}")
        print("=" * 50)
        
        # 更新配置
        config['model']['teacher_forcing_ratio'] = ratio
        config['output_dir'] = f"./checkpoints/rnn_tf_{ratio}"
        
        # 创建训练器
        trainer = RNNTrainer(config, src_tokenizer, tgt_tokenizer, device)
        
        # 获取数据加载器
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 训练
        trainer.train(train_loader, val_loader)
        
        # 记录结果
        results[f'tf_{ratio}'] = {
            'best_bleu': trainer.best_bleu,
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else 0,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0
        }
    
    # 打印比较结果
    print("\n训练策略比较结果:")
    print("=" * 50)
    for strategy, result in results.items():
        print(f"{strategy}: BLEU={result['best_bleu']:.4f}, "
              f"Val Loss={result['final_val_loss']:.4f}")

def main():
    """主函数"""
    # 加载配置
    with open('configs/rnn_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载分词器 - 检查是否有已保存的词汇表
    vocab_file = './data/processed/vocab.pkl'
    if os.path.exists(vocab_file):
        print("加载已保存的词汇表...")
        import pickle
        with open(vocab_file, 'rb') as f:
            src_tokenizer, tgt_tokenizer = pickle.load(f)
        print(f"中文词汇表大小: {src_tokenizer.vocab_size}")
        print(f"英文词汇表大小: {tgt_tokenizer.vocab_size}")
    else:
        print("警告：未找到已保存的词汇表文件，需要先运行数据预处理")
        print("运行: python preprocess.py")
        return
    
    # 根据配置选择实验类型
    experiment_type = config.get('experiment', 'basic_training')
    
    if experiment_type == 'basic_training':
        # 基础训练
        trainer = RNNTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        trainer.train(train_loader, val_loader)
    
    elif experiment_type == 'compare_attention_mechanisms':
        # 比较注意力机制
        compare_attention_mechanisms(config, src_tokenizer, tgt_tokenizer, device)
    
    elif experiment_type == 'compare_training_strategies':
        # 比较训练策略
        compare_training_strategies(config, src_tokenizer, tgt_tokenizer, device)
    
    elif experiment_type == 'quick_test':
        # 快速测试（少量周期）
        original_epochs = config['training']['num_epochs']
        config['training']['num_epochs'] = 3  # 只训练3个周期
        
        trainer = RNNTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        trainer.train(train_loader, val_loader)
        
        config['training']['num_epochs'] = original_epochs

if __name__ == "__main__":
    main()