# Transformer / GPT 从零学习路径

一套面向本地自学的中文学习计划，目标是把下面这条链路真正学通并且跑通：

`RNN / LSTM -> Seq2Seq -> Attention -> Transformer -> GPT -> 预训练 -> 后训练 -> 推理`

这份资料特别适合：

- 想从零理解 ChatGPT / GPT / Transformer 基本架构的人
- 想结合论文、开源仓库和本地 demo 学习的人
- 想在 Apple Silicon Mac 上做轻量实验的人

## 仓库内容

- [00-总览-Transformer-GPT-本地学习计划](notes/00-总览-Transformer-GPT-本地学习计划.md)
  这是一份完整主线，包括先修知识、论文、仓库、公式、Mac 本地实验建议，以及预训练 / 后训练 / 推理三种工作状态的区分。

- [01-学习会话记录模板](notes/01-学习会话记录模板.md)
  这是持续学习时使用的模板，用来记录每天看的论文、跑的代码、关键公式、loss、sample 和未解决问题。

- [02-实验记录索引](notes/02-实验记录索引.md)
  这是实验总索引，用来统一管理 tokenizer、最小 GPT、训练参数和推理采样实验。

## 这套资料的特点

- 用中文写
- 保留论文原始链接
- 同时覆盖理论、代码、实验三条线
- 不把重点只放在“会调库”，而是强调理解模型为什么这样工作
- 默认以本地小模型实验为主，而不是一开始就追求大规模训练

## 学习主线

1. 补神经网络、CNN、RNN、LSTM 和 Seq2Seq 前置知识
2. 理解 Bahdanau Attention 为什么会出现
3. 读 Transformer 原始论文和配套实现
4. 学 tokenizer 和最小 GPT
5. 理解 GPT-1 / GPT-2 / Scaling Laws
6. 用 `LLMs-from-scratch` 和 `nanoGPT` 做本地实验
7. 区分预训练、后训练和推理

## 推荐使用方式

### 方式一：作为普通 Markdown 阅读

直接从 [00-总览-Transformer-GPT-本地学习计划](notes/00-总览-Transformer-GPT-本地学习计划.md) 开始阅读，然后按顺序推进。

### 方式二：作为 Obsidian 学习库

如果你使用 Obsidian，可以把整个仓库作为一个 vault 或者子目录导入，配合 `01` 和 `02` 两个文件持续记录学习和实验进度。

## 适合的本地设备

这套计划默认面向“能本地做小实验，但不追求大规模训练”的环境，例如：

- Apple Silicon MacBook Pro
- 16GB~32GB 统一内存
- PyTorch MPS

它适合：

- 最小 tokenizer 实验
- 最小 GPT 训练
- `tinyshakespeare` 级别实验
- 采样参数对比

它不以“大模型全量预训练复现”为目标。

## 后续计划

- 增加实验样例页
- 增加更细的 repo 对照关系
- 增加更多“公式 -> 代码位置 -> 实验现象”的映射笔记

## 版本

当前版本：`v0.1.0`

