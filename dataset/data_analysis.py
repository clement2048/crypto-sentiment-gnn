"""统计数据集中各个标签的数量分布"""
import sys
import os
# 把项目根目录加入 path，否则可能找不到 data 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import Counter
from data.loader import load_posts
from data.block_builder import build_comment_blocks
from data.temporal_split import temporal_split_blocks



def count_dataset_labels(filepath: str = "dataset/final.jsonl"):
    # 1. 加载所有帖子
    posts = load_posts(filepath)
    
    # 2. 拆解成 CommentBlock 样本
    blocks, issues = build_comment_blocks(posts)
    
    # 3. 统计各个 label 的数量
    # 通常 1 代表看涨(正样本)，-1 代表看跌(负样本)，0 代表中立
    counts = Counter(block.label for block in blocks)
    
    print(f"统计文件: {filepath}")
    print(f"总产生的 Block 样本数: {len(blocks)}")
    print(f"------------------------")
    print(f"看涨 正样本 (label= 1): {counts.get(1, 0)} 条")
    print(f"看跌 负样本 (label=-1): {counts.get(-1, 0)} 条")
    print(f"中立 中样本 (label= 0): {counts.get(0, 0)} 条")




def print_split_stats(name: str, blocks: list):
    """辅助函数：打印分割集的统计信息"""
    counts = Counter(b.label for b in blocks)
    total = len(blocks)
    pos = counts.get(1, 0)
    neg = counts.get(-1, 0)
    neu = counts.get(0, 0)
    
    # 避免分母为 0 报错
    pos_rate = (pos / total * 100) if total > 0 else 0
    neg_rate = (neg / total * 100) if total > 0 else 0
    
    print(f"=== {name} 集 (Train/Val/Test) ===")
    print(f"总样本数: {total}")
    print(f"  🟢 正样本 (1) : {pos} 条 ({pos_rate:.1f}%)")
    print(f"  🔴 负样本 (-1): {neg} 条 ({neg_rate:.1f}%)")
    if neu > 0:
        print(f"  ⚪ 中立样本 (0): {neu} 条")
    print("-" * 30)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_dir, "final.jsonl")
    print(f"正在加载并分析数据集: {filepath}...\n")
    
    # 1. 加载和构建基础 Blocks
    posts = load_posts(filepath)
    blocks, issues = build_comment_blocks(posts)
    
    print(f"【整体总览】")
    print(f"成功拆解出 Block 总数: {len(blocks)}")
    print(f"过滤掉的错误数量: {len(issues)}\n")
    
    # 2. 调用现有的切分逻辑进行切分 (按时间切分)
    # 这与你执行 full 或 evaluate 参数时跑的切分是一模一样的
    splits = temporal_split_blocks(blocks)
    
    # 3. 分别统计打印
    print_split_stats("Train (训练)", splits.train)
    print_split_stats("Validation (验证)", splits.val)
    print_split_stats("Test (测试)", splits.test)

if __name__ == "__main__":
    main()