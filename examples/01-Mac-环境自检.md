# Demo 01：Mac 环境自检

## 目标

- 确认 Python / uv / PyTorch / MPS 能正常使用
- 为后面的 tokenizer、最小 GPT 和 nanoGPT 小实验打底

## 参考文档

- [Apple: Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/)
- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps)
- [uv Installation](https://docs.astral.sh/uv/getting-started/installation/)

## 步骤

```bash
xcode-select --install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
mkdir -p ~/llm-lab && cd ~/llm-lab
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision torchaudio jupyter matplotlib numpy pandas sentencepiece tiktoken notebook
```

## 验证脚本

```bash
python ../scripts/verify_mps.py
```

## 成功标准

- 能输出 Python 版本
- 能输出 `torch` 版本
- `torch.backends.mps.is_available()` 为 `True`

