import yaml
from utils.preprocessor import preprocess_data
from utils.data_loader import create_data_loaders

def main():
    # 加载配置
    with open('configs/data_config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)['data_config']
    
    # 数据预处理
    src_tokenizer, tgt_tokenizer, config = preprocess_data(config)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(config, src_tokenizer, tgt_tokenizer)
    
    print("数据加载器创建完成！")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

if __name__ == "__main__":
    main()