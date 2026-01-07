import jieba
import sacrebleu
import sentencepiece as spm

class TranslationMetrics:
    """机器翻译评估指标 - 使用sacrebleu计算BLEU-4"""
    
    def __init__(self):
        self.supported_languages = ['chinese', 'english']
    
    def _tokenize_text(self, text, language='chinese'):
        """根据语言类型进行分词"""
        if language == 'chinese':
            # 中文使用jieba分词
            tokens = jieba.lcut(text)
            return [token for token in tokens if token.strip()]
        elif language == 'english':
            # 英文使用sentencepiece分词
            # 这里可以加载预训练的sentencepiece模型
            # 暂时使用空格分词，实际使用时可以加载特定模型
            tokens = text.split()
            return [token for token in tokens if token.strip()]
        else:
            # 默认使用空格分词
            tokens = text.split()
            return [token for token in tokens if token.strip()]
    
    def compute_bleu4(self, references, hypotheses, src_language='chinese', tgt_language='english'):
        """
        使用sacreBLEU计算BLEU-4分数
        Args:
            references: 参考翻译列表
            hypotheses: 模型生成翻译列表
            src_language: 源语言 ('chinese', 'english')
            tgt_language: 目标语言 ('chinese', 'english')
        """
        if len(references) != len(hypotheses):
            raise ValueError("参考翻译和生成翻译数量不匹配")
        
        if len(references) == 0:
            return 0.0
        
        # 使用语言特定的分词器进行分词
        ref_tokens_list = []
        hyp_tokens_list = []
        
        for ref, hyp in zip(references, hypotheses):
            # 使用相应的分词器
            ref_tokens = self._tokenize_text(ref, tgt_language)
            hyp_tokens = self._tokenize_text(hyp, tgt_language)
            
            ref_tokens_list.append(ref_tokens)
            hyp_tokens_list.append(hyp_tokens)
        
        # 转换为空格分隔的字符串（sacrebleu需要）
        ref_strings = [' '.join(tokens) for tokens in ref_tokens_list]
        hyp_strings = [' '.join(tokens) for tokens in hyp_tokens_list]
        
        # 根据目标语言设置sacrebleu的分词器
        if tgt_language == 'chinese':
            # 中文使用zh分词器
            bleu_score = sacrebleu.corpus_bleu(hyp_strings, [ref_strings], tokenize='zh')
        else:
            # 英文使用13a分词器（标准英语分词）
            bleu_score = sacrebleu.corpus_bleu(hyp_strings, [ref_strings], tokenize='13a')
        
        return bleu_score.score / 100.0  # 转换为0-1范围

class TextProcessor:
    """文本处理工具类"""
    
    @staticmethod
    def chinese_tokenize(text):
        """中文分词（使用jieba）"""
        return list(jieba.cut(text))
    
    @staticmethod
    def english_tokenize(text):
        """英文分词（使用sentencepiece）"""
        # 这里可以加载预训练的sentencepiece模型
        # 暂时使用空格分词
        return text.split()
    
    @staticmethod
    def tokenize_text(text, language='chinese'):
        """根据语言分词"""
        if language == 'chinese':
            return TextProcessor.chinese_tokenize(text)
        elif language == 'english':
            return TextProcessor.english_tokenize(text)
        else:
            raise ValueError(f"不支持的语言: {language}")
    
    @staticmethod
    def preprocess_for_bleu(references, hypotheses, src_language='chinese', tgt_language='english'):
        """
        为BLEU计算预处理文本
        Returns:
            tokenized_refs: 分词后的参考翻译
            tokenized_hyps: 分词后的生成翻译
        """
        tokenized_refs = []
        tokenized_hyps = []
        
        for ref, hyp in zip(references, hypotheses):
            # 分词
            ref_tokens = TextProcessor.tokenize_text(ref, tgt_language)
            hyp_tokens = TextProcessor.tokenize_text(hyp, tgt_language)
            
            tokenized_refs.append(ref_tokens)
            tokenized_hyps.append(hyp_tokens)
        
        return tokenized_refs, tokenized_hyps

def evaluate_translation(references, hypotheses, src_language='chinese', tgt_language='english'):
    """
    完整的翻译评估函数
    """
    metrics = TranslationMetrics()
    
    # 计算BLEU-4分数
    results = {}
    results['bleu'] = metrics.compute_bleu4(references, hypotheses, src_language, tgt_language)
    results['BLEU-4'] = results['bleu']
    
    return results

def evaluate_translation_fast(references, hypotheses, src_language='chinese', tgt_language='english'):
    """
    快速翻译评估函数 - 训练时使用
    """
    return evaluate_translation(references, hypotheses, src_language, tgt_language)

def evaluate_translation_standard(references, hypotheses, src_language='chinese', tgt_language='english'):
    """
    标准翻译评估函数 - 最终评估时使用
    """
    return evaluate_translation(references, hypotheses, src_language, tgt_language)

def print_evaluation_results(results, model_name="", eval_type="fast"):
    """打印评估结果"""
    if model_name:
        print(f"\n=== {model_name} {eval_type.upper()}评估结果 ===")
    
    print(f"BLEU-4: {results['bleu']:.4f}")

# 使用示例
if __name__ == "__main__":
    # 示例数据 - 中文测试
    chinese_references = [
        "今天天气很好",
        "我喜欢机器学习",
        "这是一个测试句子"
    ]
    
    chinese_hypotheses = [
        "今天天气不错",
        "我热爱机器学习",
        "这是一个测试语句"
    ]
    
    # 示例数据 - 英文测试
    english_references = [
        "The weather is very good today",
        "I like machine learning",
        "This is a test sentence"
    ]
    
    english_hypotheses = [
        "The weather is nice today",
        "I love machine learning",
        "This is a test statement"
    ]
    
    print("=== 中文翻译测试 ===")
    # 中文评估
    chinese_results = evaluate_translation(chinese_references, chinese_hypotheses, 
                                          src_language='chinese', tgt_language='chinese')
    print_evaluation_results(chinese_results, "中文模型", "standard")
    
    print("\n=== 英文翻译测试 ===")
    # 英文评估
    english_results = evaluate_translation(english_references, english_hypotheses,
                                          src_language='english', tgt_language='english')
    print_evaluation_results(english_results, "英文模型", "standard")
    
    print("\n=== 中英翻译测试 ===")
    # 中英翻译评估
    cross_results = evaluate_translation(chinese_references, english_hypotheses,
                                        src_language='chinese', tgt_language='english')
    print_evaluation_results(cross_results, "中英翻译模型", "standard")
    
    print("\n=== 英中翻译测试 ===")
    # 英中翻译评估
    cross_results2 = evaluate_translation(english_references, chinese_hypotheses,
                                         src_language='english', tgt_language='chinese')
    print_evaluation_results(cross_results2, "英中翻译模型", "standard")