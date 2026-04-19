# Demo 02：minbpe tokenizer

## 目标

- 理解 BPE tokenizer 是怎么从原始文本中合并子词的
- 观察 `vocab size` 对序列长度的影响

## 仓库

- [karpathy/minbpe](https://github.com/karpathy/minbpe)

## 建议步骤

```bash
cd ~/llm-lab/repos
git clone https://github.com/karpathy/minbpe.git
cd minbpe
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 你要记录什么

- 原始文本长度
- tokenized 后的 token 数量
- 不同 `vocab size` 下序列长度是否明显变化

## 推荐对照

- 小词表
- 中等词表
- 更大的词表

## 实验问题

1. 为什么词表变大后，通常 token 数会变少？
2. 为什么不能无限增大词表？
3. 词表大小如何影响训练成本？

