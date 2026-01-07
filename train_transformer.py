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

from models.transformer_model import create_transformer_model
from utils.data_loader import create_data_loaders
from utils.metrics import evaluate_translation_fast, evaluate_translation_standard, print_evaluation_results
from utils.tokenizer import Tokenizer

class TransformerTrainer:
    """Transformer模型训练器"""
    
    def __init__(self, config, src_tokenizer, tgt_tokenizer, device):
        """初始化训练器"""
        self.config = config
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device
        
        # 创建模型
        model_config = config['model']
        self.model = create_transformer_model(
            src_tokenizer.vocab_size,
            tgt_tokenizer.vocab_size,
            model_config
        ).to(device)
        
        # 优化器和损失函数
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.val_bleu_scores = []
        self.best_bleu = 0
        
        # 创建输出目录
        os.makedirs(config.get('output_dir', './checkpoints/transformer'), exist_ok=True)
    
    def _create_model(self):
        """创建Transformer模型"""
        model_config = self.config['model']
        
# 判断是否使用预训练模型
        use_pretrained = model_config.get('use_pretrained', False)
        
        if use_pretrained:
            print("使用预训练T5模型进行微调")
            model_name = model_config.get('pretrained_model', 't5-small')
            model = create_transformer_model(
                src_vocab_size=self.src_tokenizer.vocab_size,
                tgt_vocab_size=self.tgt_tokenizer.vocab_size,
                use_pretrained=True,
                model_name=model_name
            )
        else:
            print("从零开始训练Transformer模型")
            # 创建模型配置字典
            model_config_dict = {
                'd_model': model_config.get('d_model', 512),
                'n_layers': model_config.get('n_layers', 6),
                'n_heads': model_config.get('n_heads', 8),
                'd_ff': model_config.get('d_ff', 2048),
                'max_seq_len': model_config.get('max_seq_len', 5000),
                'dropout': model_config.get('dropout', 0.1),
                'pos_encoding': model_config.get('pos_encoding', 'absolute'),
                'norm_type': model_config.get('norm_type', 'layernorm')
            }
            
            model = create_transformer_model(
                src_vocab_size=self.src_tokenizer.vocab_size,
                tgt_vocab_size=self.tgt_tokenizer.vocab_size,
                config=model_config_dict  # 传递配置字典而不是单独的参数
            )
        
        print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
        return model
    
    def _create_optimizer(self):
        """创建优化器"""
        training_config = self.config['training']
        optimizer_type = training_config.get('optimizer', 'adam')
        lr = training_config.get('learning_rate', 0.0001)
        weight_decay = training_config.get('weight_decay', 0.0)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(), 
                lr=lr, 
                betas=(0.9, 0.98), 
                eps=1e-9,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        training_config = self.config['training']
        
        if training_config.get('scheduler') == 'warmup':
            # Transformer常用的warmup调度器
            warmup_steps = training_config.get('warmup_steps', 4000)
            
            def lr_lambda(step):
                if step == 0:
                    return 0
                return min(step ** -0.5, step * warmup_steps ** -1.5)
            
            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        elif training_config.get('scheduler') == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                factor=0.5,
                patience=3
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch} Training')
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src_ids'].to(self.device)
            tgt = batch['tgt_ids'].to(self.device)
            
            # 准备输入和目标（右移一位）
            tgt_input = tgt[:, :-1]
            tgt_target = tgt[:, 1:]
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)
            
            # 计算损失
            output = output.contiguous().view(-1, output.size(-1))
            tgt_target = tgt_target.contiguous().view(-1)
            loss = self.criterion(output, tgt_target)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（针对Transformer）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # 更新学习率（如果是warmup调度器）
            if self.scheduler and hasattr(self.scheduler, 'step'):
                self.scheduler.step()
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader, epoch, use_fast_eval=True):
        """验证方法 - 添加调试信息"""
        self.model.eval()
        total_loss = 0
        all_references = []
        all_hypotheses = []
        
        # 限制验证样本数量：训练过程中用200个样本，训练完成后用500
        max_bleu_samples = 200 if use_fast_eval else 500 #快速评估用200，标准评估用500
        processed_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch} Validation'):
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                
                # 计算验证损失
                tgt_input = tgt[:, :-1]
                tgt_target = tgt[:, 1:]
                output = self.model(src, tgt_input)
                
                output_flat = output.contiguous().view(-1, output.size(-1))
                tgt_flat = tgt_target.contiguous().view(-1)
                loss = self.criterion(output_flat, tgt_flat)
                total_loss += loss.item()
                
                # 只在需要时生成翻译用于BLEU计算
                if processed_samples < max_bleu_samples:
                    # 使用贪心解码加速
                    for i in range(min(src.size(0), max_bleu_samples - processed_samples)):
                        src_sentence = src[i:i+1]  # [1, seq_len]
                        
                        # 使用贪心解码（beam_size=1）加速
                        translation_ids = self.model.translate(
                            src_sentence, 
                            beam_size=1  # 改为贪心解码
                        )
                        
                        # 解码为文本
                        ref_text = self.tgt_tokenizer.decode(
                            tgt[i].cpu().numpy(), 
                            remove_special_tokens=True
                        )
                        hyp_text = self.tgt_tokenizer.decode(
                            translation_ids[0].cpu().numpy(), 
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
        num_epochs = training_config.get('num_epochs', 50)
        patience = training_config.get('patience', 7)
        
        # 验证频率：快速评估每epoch，标准评估只在训练完成后
        standard_eval_frequency = 5  # 每5个epoch进行一次标准评估
        
        print("开始训练Transformer模型...")
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
            
            # 更新学习率
            if self.scheduler and hasattr(self.scheduler, 'step'):
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
            
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
                # 移除保存非最佳模型的checkpoint调用
                self.save_checkpoint(epoch, is_best=False)
            
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
        self.save_training_history()

    def save_checkpoint(self, epoch, is_best=False):
        """保存模型检查点 - 只保存最佳模型"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'val_bleu': self.val_bleu_scores[-1] if self.val_bleu_scores else 0,
            'best_bleu': self.best_bleu,
            'config': self.config
        }
        
        checkpoint_dir = self.config.get('output_dir', './checkpoints/transformer')
        
        # 只保存最佳模型，不保存每个epoch的checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"保存最佳模型: epoch {epoch}, BLEU: {self.best_bleu:.4f}")

        # 移除保存常规检查点的代码
        # checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        # torch.save(checkpoint, checkpoint_path)
        
        # 移除重复保存最佳模型的代码
        # if is_best:
        #     best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        #     torch.save(checkpoint, best_path)
    
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
            self.config.get('output_dir', './checkpoints/transformer'),
            'training_history.json'
        )
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练状态
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_bleu_scores = checkpoint.get('val_bleu_scores', [])
        self.best_bleu = checkpoint.get('best_bleu', 0)
        
        print(f"加载检查点: {checkpoint_path}")
        print(f"训练周期: {checkpoint['epoch']}")
        print(f"最佳BLEU: {self.best_bleu:.4f}")

def compare_position_encodings(config, src_tokenizer, tgt_tokenizer, device):
    """比较不同位置编码方法"""
    pos_encodings = ['absolute', 'relative']
    results = {}
    
    for pos_enc in pos_encodings:
        print(f"\n训练 {pos_enc} 位置编码...")
        print("=" * 50)
        
        # 更新配置
        config['model']['pos_encoding'] = pos_enc
        config['output_dir'] = f"./checkpoints/transformer_{pos_enc}"
        # 创建训练器
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        
        # 获取数据加载器
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 训练
        trainer.train(train_loader, val_loader)
        
        # 记录结果
        results[pos_enc] = {
            'best_bleu': trainer.best_bleu,
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else 0,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0
        }
    
    # 打印比较结果
    print("\n位置编码比较结果:")
    print("=" * 50)
    for pos_enc, result in results.items():
        print(f"{pos_enc}: BLEU={result['best_bleu']:.4f}, "
              f"Val Loss={result['final_val_loss']:.4f}")

def compare_normalization_methods(config, src_tokenizer, tgt_tokenizer, device):
    """比较不同归一化方法"""
    norm_methods = ['layernorm', 'rmsnorm']
    results = {}
    
    for norm in norm_methods:
        print(f"\n训练 {norm} 归一化...")
        print("=" * 50)
        
        # 更新配置
        config['model']['norm_type'] = norm
        config['output_dir'] = f"./checkpoints/transformer_{norm}"
        
        # 创建训练器
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        
        # 获取数据加载器
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 训练
        trainer.train(train_loader, val_loader)
        
        # 记录结果
        results[norm] = {
            'best_bleu': trainer.best_bleu,
            'final_train_loss': trainer.train_losses[-1] if trainer.train_losses else 0,
            'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else 0
        }
    
    # 打印比较结果
    print("\n归一化方法比较结果:")
    print("=" * 50)
    for norm, result in results.items():
        print(f"{norm}: BLEU={result['best_bleu']:.4f}, "
              f"Val Loss={result['final_val_loss']:.4f}")

def analyze_hyperparameter_sensitivity(config, src_tokenizer, tgt_tokenizer, device):
    """分析超参数敏感性"""
    # 测试不同的批量大小
    batch_sizes = [16, 32, 64]
    batch_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n测试批量大小: {batch_size}")
        print("=" * 50)
        
        config['data']['batch_size'] = batch_size
        config['output_dir'] = f"./checkpoints/transformer_bs_{batch_size}"
        
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 只训练少量周期用于快速测试
        original_epochs = config['training']['num_epochs']
        config['training']['num_epochs'] = 5
        trainer.train(train_loader, val_loader)
        config['training']['num_epochs'] = original_epochs
        
        batch_results[batch_size] = trainer.best_bleu
    # 测试不同的学习率
    learning_rates = [0.0001, 0.0005, 0.001]
    lr_results = {}
    
    for lr in learning_rates:
        print(f"\n测试学习率: {lr}")
        print("=" * 50)
        
        config['training']['learning_rate'] = lr
        config['output_dir'] = f"./checkpoints/transformer_lr_{lr}"
        
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 只训练少量周期用于快速测试
        original_epochs = config['training']['num_epochs']
        config['training']['num_epochs'] = 5
        trainer.train(train_loader, val_loader)
        config['training']['num_epochs'] = original_epochs
        
        lr_results[lr] = trainer.best_bleu
    
    # 测试不同的模型规模
    model_sizes = [
        {'d_model': 256, 'n_layers': 4, 'n_heads': 4, 'd_ff': 1024},
        {'d_model': 512, 'n_layers': 6, 'n_heads': 8, 'd_ff': 2048},
        {'d_model': 768, 'n_layers': 12, 'n_heads': 12, 'd_ff': 3072}
    ]
    size_results = {}
    
    for i, size_config in enumerate(model_sizes):
        print(f"\n测试模型规模 {i+1}")
        print("=" * 50)
        
        config['model'].update(size_config)
        config['output_dir'] = f"./checkpoints/transformer_size_{i+1}"
        
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 只训练少量周期用于快速测试
        original_epochs = config['training']['num_epochs']
        config['training']['num_epochs'] = 5
        trainer.train(train_loader, val_loader)
        config['training']['num_epochs'] = original_epochs
        
        size_results[f"size_{i+1}"] = {
            'bleu': trainer.best_bleu,
            'params': sum(p.numel() for p in trainer.model.parameters())
        }
    # 打印敏感性分析结果
    print("\n超参数敏感性分析结果:")
    print("=" * 50)
    print("批量大小影响:")
    for bs, bleu in batch_results.items():
        print(f"  批量大小 {bs}: BLEU={bleu:.4f}")
    
    print("\n学习率影响:")
    for lr, bleu in lr_results.items():
        print(f"  学习率 {lr}: BLEU={bleu:.4f}")
    
    print("\n模型规模影响:")
    for size, result in size_results.items():
        print(f"  {size}: BLEU={result['bleu']:.4f}, 参数量={result['params']:,}")

def compare_pretrained_vs_scratch(config, src_tokenizer, tgt_tokenizer, device):
    """比较预训练模型和从头训练"""
    strategies = [
        {'use_pretrained': False, 'name': 'scratch'},
        {'use_pretrained': True, 'name': 'pretrained_t5_small'}
    ]
    results = {}
    
    for strategy in strategies:
        print(f"\n训练策略: {strategy['name']}")
        print("=" * 50)
        
        config['model']['use_pretrained'] = strategy['use_pretrained']
        config['output_dir'] = f"./checkpoints/transformer_{strategy['name']}"
        
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        
        # 训练
        trainer.train(train_loader, val_loader)
        
        # 记录结果
        results[strategy['name']] = {
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
    with open('configs/transformer_config.yaml', 'r', encoding='utf-8') as f:
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
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        trainer.train(train_loader, val_loader)
    
    elif experiment_type == 'compare_position_encoding':
        # 比较位置编码
        compare_position_encodings(config, src_tokenizer, tgt_tokenizer, device)
    
    elif experiment_type == 'compare_normalization':
        # 比较归一化方法
        compare_normalization_methods(config, src_tokenizer, tgt_tokenizer, device)
    
    elif experiment_type == 'hyperparameter_sensitivity':
        # 超参数敏感性分析
        analyze_hyperparameter_sensitivity(config, src_tokenizer, tgt_tokenizer, device)
    
    elif experiment_type == 'compare_pretrained':
        # 比较预训练和从头训练
        compare_pretrained_vs_scratch(config, src_tokenizer, tgt_tokenizer, device)
    
    elif experiment_type == 'quick_test':
        # 快速测试（少量周期）
        original_epochs = config['training']['num_epochs']
        config['training']['num_epochs'] = 3  # 只训练3个周期
        
        trainer = TransformerTrainer(config, src_tokenizer, tgt_tokenizer, device)
        train_loader, val_loader, _ = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
        trainer.train(train_loader, val_loader)
        
        config['training']['num_epochs'] = original_epochs

if __name__ == "__main__":
    main()