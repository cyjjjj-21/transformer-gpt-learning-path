# Demo 03：microgpt 最小闭环

## 目标

- 看清 GPT 的最小训练闭环
- 把“文本 -> token -> embedding -> attention -> logits -> loss -> backward -> sample”串起来

## 资源

- [microgpt 在线版](https://karpathy.ai/microgpt.html)
- [microgpt 源码 gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

## 建议步骤

1. 先通读源码，不急着跑
2. 标出四段关键代码：
   - attention
   - loss
   - optimizer
   - sampling
3. 再运行最小实验

## 推荐记录

- 哪一段是 tokenizer
- 哪一段是参数初始化
- 哪一段是前向传播
- 哪一段是反向传播
- 哪一段是采样输出

## 实验问题

1. 为什么这个最小实现仍然能体现 GPT 的核心机制？
2. 为什么它虽然“小”，但已经包含训练和推理的完整闭环？

