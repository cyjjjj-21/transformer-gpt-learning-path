# Demo 04：nanoGPT + tinyshakespeare

## 目标

- 在本地跑一个真正可训练的小型 GPT 实验
- 学会观察 `train loss / val loss / sample` 三件事

## 仓库

- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT)

## 建议步骤

```bash
cd ~/llm-lab/repos
git clone https://github.com/karpathy/nanoGPT.git
cd nanoGPT
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python data/shakespeare_char/prepare.py
```

## 适合本地的小配置

- `block_size=64 / 128 / 256`
- `n_layer=4~6`
- `n_head=4~6`
- `n_embd=128~384`

## 建议实验

1. 先跑一个 baseline
2. 只改一个变量：
   - learning rate
   - block size
   - depth
3. 比较 sample 风格变化

## 你要记录什么

- 训练步数
- train loss
- val loss
- 是否过拟合
- sample 输出摘要

